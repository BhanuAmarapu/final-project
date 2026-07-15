from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
import os
import pymysql
from werkzeug.utils import secure_filename
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from config import Config
from ml_model import MLModel, get_ext_code
from dedup import Deduplicator
from auditing import Auditor
from utils import log_action
from suspicious_upload_detector import SuspiciousUploadDetector
from content_moderator import ContentModerator
from content_similarity import ContentSimilarityDetector, detect_similar_content
from whisper_service import whisper_service
from sentencebert_service import sentencebert_service
from similarity_service import similarity_service

app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = Config.SECRET_KEY

# Initialize modules
ml_model = MLModel()
deduplicator = Deduplicator()
auditor = Auditor()
suspicious_detector = SuspiciousUploadDetector()

# Pre-warm SBERT models and load caches on server start
print("[INIT] Pre-warming ML models and caches...")
moderator = ContentModerator()
similarity_detector = ContentSimilarityDetector()
print("[INIT] Pre-warming Audio AI models (Whisper, Sentence-BERT)...")
whisper_service.load_model()
sentencebert_service.load_model()
print("[INIT] Pre-warming complete!")

# Auto-initialize video records table on startup and migrate audio status column
try:
    print("[INIT] Checking/initializing video_records database table...")
    from mysql_wrapper import get_mysql_connection
    conn_init = get_mysql_connection()
    conn_init.execute("""
        CREATE TABLE IF NOT EXISTS video_records (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            original_filename VARCHAR(500) NOT NULL,
            uuid_filename VARCHAR(255) UNIQUE NOT NULL,
            s3_object_key VARCHAR(1000) NOT NULL,
            transcript LONGTEXT DEFAULT NULL,
            embedding LONGTEXT DEFAULT NULL,
            language VARCHAR(50) DEFAULT NULL,
            duration FLOAT DEFAULT NULL,
            similarity_score FLOAT DEFAULT NULL,
            status VARCHAR(50) DEFAULT 'completed',
            matched_file VARCHAR(500) DEFAULT NULL,
            matched_transcript LONGTEXT DEFAULT NULL,
            upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)
    conn_init.commit()
    
    # Check if 'status' column exists in audio_records, if not add it
    cursor_cols = conn_init.execute("SHOW COLUMNS FROM audio_records LIKE 'status'")
    if not cursor_cols.fetchone():
        print("[INIT] Status column missing from audio_records table. Migrating...")
        conn_init.execute("ALTER TABLE audio_records ADD COLUMN status VARCHAR(50) DEFAULT 'completed'")
        conn_init.commit()
        print("[INIT] Status column added to audio_records table.")
        
    # Check if 'matched_file' column exists, if not add it
    cursor_cols = conn_init.execute("SHOW COLUMNS FROM audio_records LIKE 'matched_file'")
    if not cursor_cols.fetchone():
        print("[INIT] matched_file column missing from audio_records table. Migrating...")
        conn_init.execute("ALTER TABLE audio_records ADD COLUMN matched_file VARCHAR(500) DEFAULT NULL")
        conn_init.execute("ALTER TABLE audio_records ADD COLUMN matched_transcript LONGTEXT DEFAULT NULL")
        conn_init.commit()
        print("[INIT] matched_file and matched_transcript columns added to audio_records table.")

    # Check if 'status' column exists in video_records, if not add all similarity columns
    cursor_cols = conn_init.execute("SHOW COLUMNS FROM video_records LIKE 'status'")
    if not cursor_cols.fetchone():
        print("[INIT] Video records table missing similarity columns. Migrating...")
        conn_init.execute("ALTER TABLE video_records ADD COLUMN transcript LONGTEXT DEFAULT NULL")
        conn_init.execute("ALTER TABLE video_records ADD COLUMN embedding LONGTEXT DEFAULT NULL")
        conn_init.execute("ALTER TABLE video_records ADD COLUMN language VARCHAR(50) DEFAULT NULL")
        conn_init.execute("ALTER TABLE video_records ADD COLUMN duration FLOAT DEFAULT NULL")
        conn_init.execute("ALTER TABLE video_records ADD COLUMN similarity_score FLOAT DEFAULT NULL")
        conn_init.execute("ALTER TABLE video_records ADD COLUMN status VARCHAR(50) DEFAULT 'completed'")
        conn_init.execute("ALTER TABLE video_records ADD COLUMN matched_file VARCHAR(500) DEFAULT NULL")
        conn_init.execute("ALTER TABLE video_records ADD COLUMN matched_transcript LONGTEXT DEFAULT NULL")
        conn_init.commit()
        print("[INIT] Similarity columns added to video_records table.")
        
    conn_init.close()
    print("[INIT] Database verification for video and audio tables complete.")
except Exception as e:
    print(f"[INIT] Warning: Could not run startup database updates: {e}")


login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id, username, role):
        self.id = id
        self.username = username
        self.role = role

@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    if user:
        return User(user['id'], user['username'], user['role'])
    return None

def get_db_connection():
    from mysql_wrapper import get_mysql_connection
    return get_mysql_connection()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password)).fetchone()
        conn.close()
        if user:
            user_obj = User(user['id'], user['username'], user['role'])
            login_user(user_obj)
            return redirect(url_for('dashboard'))
        flash('Invalid credentials')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form.get('email', 'no_email@example.com')
        password = request.form['password']
        role = request.form.get('role', 'user')
        conn = get_db_connection()
        try:
            conn.execute("INSERT INTO users (username, email, password, role) VALUES (?, ?, ?, ?)", (username, email, password, role))
            conn.commit()
            flash('Registration successful. Please login.')
            return redirect(url_for('login'))
        except pymysql.err.IntegrityError:
            flash('Username already exists')
        conn.close()
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file:
            filename = secure_filename(file.filename)
            temp_path = os.path.join(app.config['UPLOAD_TEMP'], filename)
            file.save(temp_path)
            
            # Check if file is audio or video (extension-based check)
            SUPPORTED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'aac', 'flac', 'm4a'}
            SUPPORTED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'wmv', 'flv', 'ogg'}
            file_ext = filename.split('.')[-1].lower() if '.' in filename else ''
            
            if file_ext in SUPPORTED_AUDIO_EXTENSIONS:
                print(f"\n========== STARTING ASYNC AUDIO UPLOAD PIPELINE ==========")
                print(f"File: {filename}")
                log_action("Audio Upload Started", f"File: {filename}")
                
                try:
                    # 1. Validate the audio file
                    log_action("Audio Validation Passed", f"File: {filename}")
                    
                    # Generate a unique temp filename to avoid collision in UPLOAD_TEMP
                    import uuid
                    unique_temp_filename = f"audio_temp_{uuid.uuid4().hex}.{file_ext}"
                    unique_temp_path = os.path.join(app.config['UPLOAD_TEMP'], unique_temp_filename)
                    os.rename(temp_path, unique_temp_path)
                    
                    # Generate S3 key/uuid filename
                    s3_filename = f"audio_{uuid.uuid4().hex}.{file_ext}"
                    
                    # Store placeholder record in audio_records DB table immediately
                    conn = get_db_connection()
                    conn.execute("""
                        INSERT INTO audio_records 
                        (user_id, original_filename, uuid_filename, transcript, embedding, language, duration, s3_object_key, similarity_score, status)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        current_user.id,
                        filename,
                        s3_filename,
                        "Processing...",
                        "[]",
                        "en",
                        0.0,
                        s3_filename,
                        None,
                        "processing"
                    ))
                    conn.commit()
                    
                    # Fetch database record ID
                    row = conn.execute("SELECT id FROM audio_records WHERE uuid_filename = ?", (s3_filename,)).fetchone()
                    record_id = row['id'] if row else None
                    conn.close()
                    
                    if not record_id:
                        raise RuntimeError("Failed to retrieve created audio record ID")
                    
                    # Spawn daemon background thread to run evaluation and storage
                    import threading
                    
                    def run_async_pipeline(rec_id, t_path, orig_name, s3_name):
                        try:
                            # 1. Upload directly to S3 first so it is stored in S3 immediately!
                            from utils import upload_to_s3
                            if not upload_to_s3(t_path, s3_name):
                                raise RuntimeError("S3 upload failed")
                            print(f"[AsyncAudio] Uploaded {orig_name} to S3 immediately.")
                            
                            # 2. Convert speech into text using Whisper
                            from whisper_service import whisper_service
                            transcription_result = whisper_service.transcribe(t_path)
                            transcript = transcription_result["transcript"]
                            language = transcription_result["language"]
                            duration = transcription_result["duration"]
                            
                            # 3. Generate semantic embeddings using Sentence-BERT
                            from sentencebert_service import sentencebert_service
                            embedding = sentencebert_service.generate_embedding(transcript)
                            
                            # 4. Compare embedding against stored transcript embeddings (excluding ourselves!)
                            from similarity_service import similarity_service
                            similarity_result = similarity_service.find_highest_similarity(embedding, exclude_id=rec_id)
                            similarity_score = similarity_result["similarity"]
                            
                            # If similarity is >= 60%, status is pending_confirmation, otherwise completed
                            status = "pending_confirmation" if similarity_score >= 0.60 else "completed"
                            
                            # Retrieve matches
                            matched_record = similarity_result.get("matched_record")
                            matched_filename = matched_record["original_filename"] if matched_record else None
                            matched_transcript = matched_record["transcript"] if matched_record else None
                            
                            # 5. Update database record with final values
                            import json
                            conn_thread = get_db_connection()
                            conn_thread.execute("""
                                UPDATE audio_records 
                                SET transcript = ?, embedding = ?, language = ?, duration = ?, 
                                    similarity_score = ?, status = ?, matched_file = ?, matched_transcript = ?
                                WHERE id = ?
                            """, (
                                transcript,
                                json.dumps(embedding),
                                language,
                                duration,
                                similarity_score,
                                status,
                                matched_filename,
                                matched_transcript,
                                rec_id
                            ))
                            conn_thread.commit()
                            conn_thread.close()
                            print(f"[AsyncAudio] Processed and finalized {orig_name} successfully with status {status}.")
                            
                        except Exception as thread_err:
                            print(f"[AsyncAudio] Error processing audio {orig_name}: {thread_err}")
                            try:
                                conn_thread = get_db_connection()
                                conn_thread.execute("""
                                    UPDATE audio_records 
                                    SET transcript = ?, similarity_score = 0.0, status = 'failed'
                                    WHERE id = ?
                                """, (f"Processing failed: {str(thread_err)}", rec_id))
                                conn_thread.commit()
                                conn_thread.close()
                            except Exception as db_err:
                                print(f"[AsyncAudio] Error updating failure status: {db_err}")
                        finally:
                            # Cleanup local temp file
                            if os.path.exists(t_path):
                                try:
                                    os.remove(t_path)
                                except Exception as clean_err:
                                    print(f"[AsyncAudio] Error deleting temp file: {clean_err}")
                                    
                    threading.Thread(
                        target=run_async_pipeline,
                        args=(record_id, unique_temp_path, filename, s3_filename),
                        daemon=True
                    ).start()
                    
                    return jsonify({
                        "status": "processing",
                        "audio_id": record_id,
                        "message": "Audio file uploaded and is being processed in the background."
                    })
                        
                except Exception as e:
                    log_action("Upload Error", f"Error in audio upload pipeline: {str(e)}")
                    # Try cleaning up unique temp path if it exists
                    if 'unique_temp_path' in locals() and os.path.exists(unique_temp_path):
                        try:
                            os.remove(unique_temp_path)
                        except:
                            pass
                    return jsonify({"error": f"Audio processing failed: {str(e)}"}), 500
                    
            elif file_ext in SUPPORTED_VIDEO_EXTENSIONS:
                print(f"\n========== STARTING VIDEO UPLOAD PIPELINE ==========")
                print(f"File: {filename}")
                log_action("Video Upload Started", f"File: {filename}")
                
                try:
                    # Generate unique temp filename to prevent collision in UPLOAD_TEMP
                    import uuid
                    unique_temp_filename = f"video_temp_{uuid.uuid4().hex}.{file_ext}"
                    unique_temp_path = os.path.join(app.config['UPLOAD_TEMP'], unique_temp_filename)
                    os.rename(temp_path, unique_temp_path)
                    
                    # Generate S3 key/uuid filename
                    s3_filename = f"video_{uuid.uuid4().hex}.{file_ext}"
                    
                    # Store placeholder in video_records DB table immediately
                    conn = get_db_connection()
                    conn.execute("""
                        INSERT INTO video_records 
                        (user_id, original_filename, uuid_filename, s3_object_key, status, transcript, embedding)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        current_user.id,
                        filename,
                        s3_filename,
                        s3_filename,
                        "processing",
                        "Processing video...",
                        "[]"
                    ))
                    conn.commit()
                    
                    # Fetch database record ID
                    row = conn.execute("SELECT id FROM video_records WHERE uuid_filename = ?", (s3_filename,)).fetchone()
                    record_id = row['id'] if row else None
                    conn.close()
                    
                    if not record_id:
                        raise RuntimeError("Failed to retrieve created video record ID")
                        
                    # Spawn background daemon thread to run audio extraction, Whisper, SBERT, and comparison
                    import threading
                    
                    def run_video_async_pipeline(rec_id, t_path, orig_name, s3_name):
                        temp_extracted_audio = None
                        try:
                            # 1. Upload video directly to S3 first
                            from utils import upload_to_s3
                            if not upload_to_s3(t_path, s3_name):
                                raise RuntimeError("S3 upload failed")
                            print(f"[AsyncVideo] Uploaded {orig_name} to S3 immediately.")
                            
                            # 2. Extract audio track using FFmpeg
                            import subprocess
                            import uuid
                            audio_filename = f"extracted_{uuid.uuid4().hex}.mp3"
                            temp_extracted_audio = os.path.join(app.config['UPLOAD_TEMP'], audio_filename)
                            
                            cmd = [
                                "ffmpeg", "-y", "-i", t_path, 
                                "-vn", "-acodec", "libmp3lame", "-ar", "16000", "-ac", "1", 
                                temp_extracted_audio
                            ]
                            
                            print(f"[AsyncVideo] Extracting audio track using command: {' '.join(cmd)}")
                            process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            
                            transcript = "No audio track detected or extraction failed."
                            language = "en"
                            duration = 0.0
                            embedding = []
                            similarity_score = 0.0
                            status = "completed"
                            matched_filename = None
                            matched_transcript = None
                            
                            # Check if audio file was successfully generated and contains actual data
                            if process.returncode == 0 and os.path.exists(temp_extracted_audio) and os.path.getsize(temp_extracted_audio) > 1000:
                                print(f"[AsyncVideo] Audio track successfully extracted. Running Whisper transcription...")
                                
                                # 3. Convert speech into text using Whisper ASR
                                from whisper_service import whisper_service
                                transcription_result = whisper_service.transcribe(temp_extracted_audio)
                                transcript = transcription_result["transcript"]
                                language = transcription_result["language"]
                                duration = transcription_result["duration"]
                                
                                if transcript.strip():
                                    # 4. Generate semantic embeddings using Sentence-BERT
                                    from sentencebert_service import sentencebert_service
                                    embedding = sentencebert_service.generate_embedding(transcript)
                                    
                                    # 5. Compare embedding against other stored video transcripts (excluding ourselves)
                                    from similarity_service import similarity_service
                                    similarity_result = similarity_service.find_highest_similarity(
                                        embedding, 
                                        exclude_id=rec_id, 
                                        table_name="video_records"
                                    )
                                    similarity_score = similarity_result["similarity"]
                                    
                                    # If similarity is >= 60%, status is pending_confirmation, otherwise completed
                                    status = "pending_confirmation" if similarity_score >= 0.60 else "completed"
                                    
                                    # Retrieve matched record
                                    matched_record = similarity_result.get("matched_record")
                                    matched_filename = matched_record["original_filename"] if matched_record else None
                                    matched_transcript = matched_record["transcript"] if matched_record else None
                                    
                            # 6. Update database record with final values
                            import json
                            conn_thread = get_db_connection()
                            conn_thread.execute("""
                                UPDATE video_records 
                                SET transcript = ?, embedding = ?, language = ?, duration = ?, 
                                    similarity_score = ?, status = ?, matched_file = ?, matched_transcript = ?
                                WHERE id = ?
                            """, (
                                transcript,
                                json.dumps(embedding),
                                language,
                                duration,
                                similarity_score,
                                status,
                                matched_filename,
                                matched_transcript,
                                rec_id
                            ))
                            conn_thread.commit()
                            conn_thread.close()
                            print(f"[AsyncVideo] Processed and finalized video {orig_name} successfully with status {status}.")
                            
                        except Exception as thread_err:
                            print(f"[AsyncVideo] Error processing video {orig_name}: {thread_err}")
                            try:
                                conn_thread = get_db_connection()
                                conn_thread.execute("""
                                    UPDATE video_records 
                                    SET transcript = ?, similarity_score = 0.0, status = 'failed'
                                    WHERE id = ?
                                """, (f"Processing failed: {str(thread_err)}", rec_id))
                                conn_thread.commit()
                                conn_thread.close()
                            except Exception as db_err:
                                print(f"[AsyncVideo] Error updating failure status: {db_err}")
                        finally:
                            # Cleanup local temp files
                            if os.path.exists(t_path):
                                try:
                                    os.remove(t_path)
                                except:
                                    pass
                            if temp_extracted_audio and os.path.exists(temp_extracted_audio):
                                try:
                                    os.remove(temp_extracted_audio)
                                except:
                                    pass
                                    
                    threading.Thread(
                        target=run_video_async_pipeline,
                        args=(record_id, unique_temp_path, filename, s3_filename),
                        daemon=True
                    ).start()
                    
                    return jsonify({
                        "status": "processing",
                        "video_id": record_id,
                        "message": "Video file uploaded and is being processed in the background."
                    })
                    
                except Exception as e:
                    log_action("Upload Error", f"Error in video upload pipeline: {str(e)}")
                    if 'unique_temp_path' in locals() and os.path.exists(unique_temp_path):
                        try:
                            os.remove(unique_temp_path)
                        except:
                            pass
                    return jsonify({"error": f"Video processing failed: {str(e)}"}), 500

            # STEP 0: AI CONTENT MODERATION CHECK (BEFORE ANY PROCESSING)
            print(f"\n========== CONTENT MODERATION CHECK ==========")
            print(f"File: {filename}")
            
            moderation_result = moderator.moderate_file(temp_path, filename)
            
            if not moderation_result.is_safe:
                print(f"[MODERATION] [X] REJECTED: {moderation_result.violation_type}")
                print(f"[MODERATION] Details: {moderation_result.violation_details}")
                
                # Log the rejection in moderation_logs table
                try:
                    conn = get_db_connection()
                    flagged_keywords_str = ','.join(moderation_result.flagged_keywords) if moderation_result.flagged_keywords else ''
                    
                    conn.execute("""
                        INSERT INTO moderation_logs 
                        (user_id, file_name, file_type, file_size, violation_type, 
                         violation_details, confidence_score, flagged_keywords)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        current_user.id,
                        filename,
                        file.content_type or 'unknown',
                        os.path.getsize(temp_path),
                        moderation_result.violation_type,
                        moderation_result.violation_details,
                        moderation_result.confidence_score,
                        flagged_keywords_str
                    ))
                    conn.commit()
                    conn.close()
                    print(f"[MODERATION] Logged rejection to database")
                except Exception as e:
                    print(f"[MODERATION] Error logging rejection: {e}")
                
                # Create admin alert for suspicious activity
                try:
                    conn = get_db_connection()
                    alert_description = f"User attempted to upload inappropriate content: {filename}"
                    alert_details = f"Violation: {moderation_result.violation_type}\nDetails: {moderation_result.violation_details}\nConfidence: {moderation_result.confidence_score:.2%}"
                    
                    conn.execute("""
                        INSERT INTO suspicious_activities 
                        (user_id, activity_type, severity, description, details)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        current_user.id,
                        'INAPPROPRIATE_CONTENT',
                        'CRITICAL',
                        alert_description,
                        alert_details
                    ))
                    conn.commit()
                    conn.close()
                    print(f"[MODERATION] Created admin alert")
                except Exception as e:
                    print(f"[MODERATION] Error creating alert: {e}")
                
                # Delete the temp file immediately
                try:
                    os.remove(temp_path)
                    print(f"[MODERATION] Deleted temp file")
                except Exception as e:
                    print(f"[MODERATION] Error deleting temp file: {e}")
                
                # Return rejection message to user
                flash("Your upload has been rejected due to violation of content policies.", "danger")
                return redirect(url_for('upload_file'))
            
            print(f"[MODERATION] [OK] Content passed moderation check")
            
            # Step 1: Compute file hash early for exact duplicate detection
            from utils import get_file_hash
            file_hash = get_file_hash(temp_path)
            file_size = os.path.getsize(temp_path)
            ext_code = get_ext_code(filename)
            
            # Open database connection for all queries
            conn = get_db_connection()
            
            # Check for IDENTICAL files (exact hash match)
            identical_files = conn.execute("""
                SELECT id, file_name, file_size, file_hash, upload_timestamp, stored_path 
                FROM files 
                WHERE file_hash = ?
                ORDER BY upload_timestamp DESC
            """, (file_hash,)).fetchall()
            
            # Get frequency
            freq = conn.execute("SELECT COUNT(*) FROM files WHERE file_name = ?", (filename,)).fetchone()[0]
            
            # Find SIMILAR files based on metadata (excluding identical matches)
            # Only show if: exact same filename OR (very close size AND same extension AND similar name pattern)
            similar_files = conn.execute("""
                SELECT id, file_name, file_size, file_hash, upload_timestamp, stored_path 
                FROM files 
                WHERE file_hash != ? AND file_name = ?
                ORDER BY upload_timestamp DESC
                LIMIT 5
            """, (file_hash, filename)).fetchall()
            
            # Close connection after all queries
            conn.close()
            
            # ML Prediction
            prediction = ml_model.predict({
                'file_size': file_size,
                'extension_code': ext_code,
                'frequency': freq + 1
            })
            
            # IMPORTANT: Extract content BEFORE similarity detection
            # This allows the similarity detector to read the file content
            file_content_text = None
            try:
                if similarity_detector.is_text_file(filename):
                    file_content_text = similarity_detector.read_file_content(temp_path)
                    if file_content_text:
                        print(f"[DEBUG] Extracted {len(file_content_text)} characters from uploaded file")
                    else:
                        print(f"[DEBUG] Could not extract content from {filename}")
            except Exception as e:
                print(f"[DEBUG] Content extraction error: {e}")
                import traceback
                traceback.print_exc()
            
            # NEW: Content-level similarity detection (80%+ match)
            print(f"\n========== STARTING CONTENT SIMILARITY CHECK ==========")
            print(f"File: {filename}, Hash: {file_hash[:12]}")
            near_duplicate_files = []
            try:
                near_duplicate_files = similarity_detector.find_similar_files(temp_path, filename, file_hash)
                print(f"Content similarity check completed. Found {len(near_duplicate_files)} near-duplicates")
            except Exception as e:
                print(f"Content similarity detection error: {e}")
                import traceback
                traceback.print_exc()

            
            # Determine match type
            match_type = "none"
            if identical_files:
                match_type = "identical"
            elif near_duplicate_files:
                match_type = "near_duplicate"
            elif prediction == 1 or freq > 0 or similar_files:
                match_type = "similar"
            
            # If duplicates detected (identical, near-duplicate, or similar), show confirmation page
            if match_type != "none":
                return render_template('upload_confirmation.html',
                                     filename=filename,
                                     temp_path=temp_path,
                                     file_size=file_size,
                                     file_hash=file_hash,
                                     prediction=prediction,
                                     identical_files=identical_files,
                                     similar_files=similar_files,
                                     near_duplicate_files=near_duplicate_files,
                                     match_type=match_type,
                                     ml_confidence="High" if prediction == 1 else "Medium")


            
            # Step 2: Deduplication (if user confirms or no duplicates predicted)
            is_duplicate, file_id = deduplicator.process_file(temp_path, filename, current_user.id)
            
            # Store content in uploads table for similarity detection
            if file_content_text and file_id:
                try:
                    conn = get_db_connection()
                    conn.execute("""
                        UPDATE uploads 
                        SET content_text = ? 
                        WHERE file_id = ? AND user_id = ?
                    """, (file_content_text, file_id, current_user.id))
                    conn.commit()
                    conn.close()
                    print(f"[DEBUG] Stored content for file_id {file_id}")
                except Exception as e:
                    print(f"[DEBUG] Could not store content: {e}")
                    
            if file_id and similarity_detector.is_image_file(filename):
                try:
                    similarity_detector.add_dino_cache(file_id, temp_path)
                    print(f"[DEBUG] Cached DINOv2 embedding for file_id {file_id}")
                except Exception as e:
                    print(f"[DEBUG] Error caching DINOv2: {e}")
            
            # Step 3: Suspicious Activity Detection
            if Config.ENABLE_SUSPICIOUS_DETECTOR:
                # Track rapid uploads
                is_rapid, rapid_msg = suspicious_detector.track_upload(current_user.id)
                if is_rapid and rapid_msg:
                    flash(rapid_msg, 'warning')
                
                # Track duplicate attempts if this is a duplicate
                if is_duplicate:
                    # Get file hash from database
                    temp_conn = get_db_connection()
                    file_hash_row = temp_conn.execute("SELECT file_hash FROM files WHERE id = ?", (file_id,)).fetchone()
                    temp_conn.close()
                    
                    if file_hash_row:
                        is_excessive, dup_msg = suspicious_detector.track_duplicate_attempt(current_user.id, file_hash_row[0])
                        if is_excessive and dup_msg:
                            flash(dup_msg, 'danger')

            
            if is_duplicate:
                flash(f"DUPLICATE ALERT: An identical file was already found in the system. Redirecting for access mapping.")
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            return render_template('results.html', 
                                   filename=filename, 
                                   prediction=prediction, 
                                   is_duplicate=is_duplicate,
                                   file_id=file_id)
            
    return render_template('upload.html')

@app.route('/confirm_upload', methods=['POST'])
@login_required
def confirm_upload():
    """Handle user's decision to store or skip the file"""
    action = request.form.get('action')
    filename = request.form.get('filename')
    
    if action == 'skip':
        flash(f'Upload cancelled. File "{filename}" was not stored.', 'info')
        return redirect(url_for('upload_file'))
    
    # User chose to store - process the file
    temp_path = os.path.join(app.config['UPLOAD_TEMP'], filename)
    
    if not os.path.exists(temp_path):
        flash('File not found. Please upload again.', 'danger')
        return redirect(url_for('upload_file'))
    
    # Process the file
    is_duplicate, file_id = deduplicator.process_file(temp_path, filename, current_user.id)
    
    # Extract and store content for future similarity checks
    try:
        if similarity_detector.is_text_file(filename):
            file_content_text = similarity_detector.read_file_content(temp_path)
            if file_content_text and file_id:
                conn = get_db_connection()
                conn.execute("""
                    UPDATE uploads 
                    SET content_text = ? 
                    WHERE file_id = ? AND user_id = ?
                """, (file_content_text, file_id, current_user.id))
                conn.commit()
                conn.close()
                print(f"[DEBUG] Stored content for file_id {file_id} (confirmed upload)")
    except Exception as e:
        print(f"[DEBUG] Could not extract/store content: {e}")
        
    if file_id and similarity_detector.is_image_file(filename):
        try:
            similarity_detector.add_dino_cache(file_id, temp_path)
            print(f"[DEBUG] Cached DINOv2 embedding for file_id {file_id} after confirmation")
        except Exception as e:
            print(f"[DEBUG] Error caching DINOv2: {e}")
    
    # Track suspicious activity
    if Config.ENABLE_SUSPICIOUS_DETECTOR:
        is_rapid, rapid_msg = suspicious_detector.track_upload(current_user.id)
        if is_rapid and rapid_msg:
            flash(rapid_msg, 'warning')
        
        if is_duplicate:
            temp_conn = get_db_connection()
            file_hash_row = temp_conn.execute("SELECT file_hash FROM files WHERE id = ?", (file_id,)).fetchone()
            temp_conn.close()
            
            if file_hash_row:
                is_excessive, dup_msg = suspicious_detector.track_duplicate_attempt(current_user.id, file_hash_row[0])
                if is_excessive and dup_msg:
                    flash(dup_msg, 'danger')
    
    if is_duplicate:
        flash(f"File stored successfully! Duplicate detected - linked to existing file.", 'success')
    else:
        flash(f'File "{filename}" uploaded and encrypted successfully!', 'success')
    
    # Clean up temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    return redirect(url_for('dashboard'))


@app.route('/confirm_audio_upload', methods=['POST'])
@login_required
def confirm_audio_upload():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid request payload"}), 400
            
        action = data.get("action")
        temp_filename = data.get("temp_filename")
        original_filename = data.get("original_filename")
        transcript = data.get("transcript")
        embedding = data.get("embedding")
        language = data.get("language")
        duration = data.get("duration")
        similarity_score = data.get("similarity_score")
        
        if not temp_filename:
            return jsonify({"error": "Missing temp_filename"}), 400
            
        temp_path = os.path.join(app.config['UPLOAD_TEMP'], temp_filename)
        
        if action == 'cancel':
            log_action("Upload Cancelled", f"File: {original_filename}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({"status": "cancelled", "message": "Upload Cancelled."})
            
        elif action == 'store':
            if not os.path.exists(temp_path):
                return jsonify({"error": "Temporary file not found. Please upload again."}), 404
                
            log_action("Upload to AWS S3 Started", f"File: {original_filename}")
            
            # S3 key (UUID filename)
            file_ext = temp_filename.split('.')[-1]
            import uuid
            s3_filename = f"audio_{uuid.uuid4().hex}.{file_ext}"
            
            # Upload directly to S3
            from utils import upload_to_s3
            if upload_to_s3(temp_path, s3_filename):
                log_action("Upload Completed", f"File: {original_filename} uploaded to S3 as {s3_filename}")
                
                # Store in DB
                import json
                conn = get_db_connection()
                conn.execute("""
                    INSERT INTO audio_records 
                    (user_id, original_filename, uuid_filename, transcript, embedding, language, duration, s3_object_key, similarity_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    current_user.id,
                    original_filename,
                    s3_filename,
                    transcript,
                    json.dumps(embedding),
                    language,
                    duration,
                    s3_filename,
                    similarity_score / 100.0 if similarity_score else None
                ))
                conn.commit()
                conn.close()
                
                # Cleanup local temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
                return jsonify({
                    "status": "success",
                    "message": "Audio stored successfully!"
                })
            else:
                log_action("AWS S3 Error", f"S3 upload failed for {original_filename}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return jsonify({"error": "AWS S3 upload failed"}), 500
        else:
            return jsonify({"error": "Invalid action"}), 400
            
    except Exception as e:
        log_action("Upload Error", f"Error in confirming audio upload: {str(e)}")
        return jsonify({"error": f"Error: {str(e)}"}), 500


@app.route('/dashboard')
@login_required
def dashboard():
    conn = get_db_connection()
    files = conn.execute("SELECT * FROM files").fetchall()
    total_files = len(files)
    
    logical_size = conn.execute("""
        SELECT SUM(f.file_size) 
        FROM uploads u 
        JOIN files f ON u.file_id = f.id
    """).fetchone()[0] or 0
    
    physical_size = conn.execute("SELECT SUM(file_size) FROM files").fetchall()[0][0] or 0
    
    dedup_rate = 0
    if logical_size > 0:
        dedup_rate = ((logical_size - physical_size) / logical_size) * 100
        
    audit_logs = conn.execute("SELECT a.*, f.file_name FROM audits a JOIN files f ON a.file_id = f.id ORDER BY a.timestamp DESC LIMIT 10").fetchall()
    
    # Load audio records
    audio_records = conn.execute("SELECT * FROM audio_records ORDER BY upload_timestamp DESC").fetchall()
    
    # Load video records
    video_records = conn.execute("SELECT * FROM video_records ORDER BY upload_timestamp DESC").fetchall()
    
    conn.close()
    
    return render_template('dashboard.html', 
                           files=files, 
                           total_files=total_files,
                           logical_size=logical_size,
                           physical_size=physical_size,
                           dedup_rate=round(dedup_rate, 2),
                           audit_logs=audit_logs,
                           audio_records=audio_records,
                           video_records=video_records)

@app.route('/admin/moderation')
@login_required
def moderation_panel():
    """Admin panel to view content moderation logs"""
    if current_user.role != 'admin':
        flash('Permission denied. Admin access required.', 'danger')
        return redirect(url_for('dashboard'))
    
    conn = get_db_connection()
    
    # Get filter parameters
    show_reviewed = request.args.get('reviewed', 'false') == 'true'
    
    # Build query
    if show_reviewed:
        moderation_logs = conn.execute("""
            SELECT m.*, u.username, u.email 
            FROM moderation_logs m
            JOIN users u ON m.user_id = u.id
            ORDER BY m.timestamp DESC
        """).fetchall()
    else:
        moderation_logs = conn.execute("""
            SELECT m.*, u.username, u.email 
            FROM moderation_logs m
            JOIN users u ON m.user_id = u.id
            WHERE m.reviewed = 0
            ORDER BY m.timestamp DESC
        """).fetchall()
    
    # Get statistics
    total_rejections = conn.execute("SELECT COUNT(*) FROM moderation_logs").fetchone()[0]
    unreviewed_count = conn.execute("SELECT COUNT(*) FROM moderation_logs WHERE reviewed = 0").fetchone()[0]
    
    conn.close()
    
    return render_template('moderation.html',
                         moderation_logs=moderation_logs,
                         total_rejections=total_rejections,
                         unreviewed_count=unreviewed_count,
                         show_reviewed=show_reviewed)

@app.route('/admin/moderation/<int:log_id>/review', methods=['POST'])
@login_required
def review_moderation(log_id):
    """Mark a moderation log as reviewed"""
    if current_user.role != 'admin':
        flash('Permission denied.', 'danger')
        return redirect(url_for('dashboard'))
    
    notes = request.form.get('notes', '')
    
    conn = get_db_connection()
    conn.execute("""
        UPDATE moderation_logs 
        SET reviewed = 1, reviewer_notes = ?
        WHERE id = ?
    """, (notes, log_id))
    conn.commit()
    conn.close()
    
    flash('Moderation log marked as reviewed.', 'success')
    return redirect(url_for('moderation_panel'))

@app.route('/audit/<int:file_id>')
@login_required
def audit(file_id):
    success, message = auditor.audit_file(file_id)
    flash(message)
    return redirect(url_for('dashboard'))

@app.route('/view/<int:file_id>')
@login_required
def view_file(file_id):
    if current_user.role != 'admin':
        flash('Permission denied. Privileged access required.')
        return redirect(url_for('dashboard'))
    
    conn = get_db_connection()
    file_data = conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
    conn.close()
    
    if not file_data:
        flash('File not found')
        return redirect(url_for('dashboard'))
    
    stored_path = file_data['stored_path']
    filename = file_data['file_name']
    
    if stored_path.startswith("s3://"):
        # Stream from S3
        from utils import get_s3_client, decrypt_file
        s3 = get_s3_client()
        if not s3:
            flash("S3 service is not available.")
            return redirect(url_for('dashboard'))
            
        s3_object_name = stored_path.split("/")[-1]
        temp_encrypted_path = os.path.join(Config.UPLOAD_TEMP, f"enc_{s3_object_name}")
        temp_decrypted_path = os.path.join(Config.UPLOAD_TEMP, f"view_{filename}")
        
        try:
            # Download encrypted file from S3
            s3.download_file(Config.S3_BUCKET_NAME, s3_object_name, temp_encrypted_path)
            
            # Decrypt it
            decrypt_file(temp_encrypted_path, temp_decrypted_path, None)
            
            def generate():
                with open(temp_decrypted_path, 'rb') as f:
                    yield from f
                # Cleanup after streaming
                if os.path.exists(temp_encrypted_path):
                    os.remove(temp_encrypted_path)
                if os.path.exists(temp_decrypted_path):
                    os.remove(temp_decrypted_path)

            from flask import Response
            return Response(generate(), mimetype='application/octet-stream',
                            headers={"Content-Disposition": f"attachment;filename={filename}"})
            
        except Exception as e:
            flash(f"Error fetching or decrypting from S3: {e}")
            if os.path.exists(temp_encrypted_path):
                os.remove(temp_encrypted_path)
            return redirect(url_for('dashboard'))
    else:
        # Local file
        # In this system, local files are encrypted, so we need to decrypt for viewing
        temp_view_path = os.path.join(Config.UPLOAD_TEMP, f"view_{filename}")
        from utils import decrypt_file
        try:
            decrypt_file(stored_path, temp_view_path, None) # Uses default key logic
            
            def generate():
                with open(temp_view_path, 'rb') as f:
                    yield from f
                if os.path.exists(temp_view_path):
                    os.remove(temp_view_path)

            from flask import Response
            return Response(generate(), mimetype='application/octet-stream',
                            headers={"Content-Disposition": f"attachment;filename={filename}"})
        except Exception as e:
            flash(f"Error decrypting file: {e}")
            return redirect(url_for('dashboard'))

@app.route('/delete/<int:file_id>', methods=['POST'])
@login_required
def delete_file(file_id):
    if current_user.role != 'admin':
        flash('Permission denied.')
        return redirect(url_for('dashboard'))
    
    conn = get_db_connection()
    file_data = conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
    
    if file_data:
        stored_path = file_data['stored_path']
        file_name = file_data['file_name']
        
        try:
            # 1. Delete from S3 if applicable
            if stored_path.startswith("s3://"):
                from utils import get_s3_client
                s3 = get_s3_client()
                if not s3:
                    flash("Warning: S3 service is not available for deletion.")
                else:
                    s3_key = stored_path.replace(f"s3://{Config.S3_BUCKET_NAME}/", "")
                    try:
                        s3.delete_object(Bucket=Config.S3_BUCKET_NAME, Key=s3_key)
                        log_action("Delete", f"Deleted {file_name} from S3")
                    except Exception as e:
                        flash(f"Warning: S3 deletion failed ({e}), proceed with DB cleanup.")
            elif os.path.exists(stored_path):
                # Delete local file
                os.remove(stored_path)
                log_action("Delete", f"Deleted {file_name} from local storage")

            # 2. Delete database records in order of dependency
            conn.execute("DELETE FROM uploads WHERE file_id = ?", (file_id,))
            conn.execute("DELETE FROM audits WHERE file_id = ?", (file_id,))
            conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
            conn.commit()
            flash(f"Success: File '{file_name}' and all associated records deleted.")
        except Exception as e:
            conn.rollback()
            flash(f"Error during deletion: {str(e)}")
        finally:
            conn.close()
    else:
        conn.close()
        flash("Error: File record not found in database.")
    
    return redirect(url_for('dashboard'))

@app.route('/rename/<int:file_id>', methods=['POST'])
@login_required
def rename_file(file_id):
    if current_user.role != 'admin':
        flash('Permission denied.')
        return redirect(url_for('dashboard'))
    
    new_name = request.form.get('new_name')
    if not new_name:
        flash("New name cannot be empty.")
        return redirect(url_for('dashboard'))
    
    conn = get_db_connection()
    conn.execute("UPDATE files SET file_name = ? WHERE id = ?", (new_name, file_id))
    conn.commit()
    conn.close()
    
    flash(f"File renamed to '{new_name}'.")
    return redirect(url_for('dashboard'))

@app.route('/audio/open/<int:audio_id>')
@login_required
def open_audio(audio_id):
    conn = get_db_connection()
    audio = conn.execute("SELECT * FROM audio_records WHERE id = ?", (audio_id,)).fetchone()
    conn.close()
    
    if not audio:
        flash('Audio file not found.')
        return redirect(url_for('dashboard'))
        
    if audio['user_id'] != current_user.id and current_user.role != 'admin':
        flash('Permission denied.')
        return redirect(url_for('dashboard'))
        
    s3_key = audio['s3_object_key']
    filename = audio['original_filename']
    
    from utils import get_s3_client
    s3 = get_s3_client()
    if not s3:
        flash("S3 service is not available.")
        return redirect(url_for('dashboard'))
        
    temp_path = os.path.join(Config.UPLOAD_TEMP, f"open_{s3_key}")
    try:
        s3.download_file(Config.S3_BUCKET_NAME, s3_key, temp_path)
        
        ext = filename.split('.')[-1].lower()
        mimetype = 'audio/mpeg'
        if ext == 'wav': mimetype = 'audio/wav'
        elif ext == 'ogg': mimetype = 'audio/ogg'
        elif ext == 'aac': mimetype = 'audio/aac'
        elif ext == 'flac': mimetype = 'audio/x-flac'
        elif ext == 'm4a': mimetype = 'audio/mp4'
        
        def generate():
            with open(temp_path, 'rb') as f:
                yield from f
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        from flask import Response
        return Response(generate(), mimetype=mimetype,
                        headers={"Content-Disposition": f"inline;filename={filename}"})
    except Exception as e:
        flash(f"Error opening audio: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return redirect(url_for('dashboard'))

@app.route('/audio/rename/<int:audio_id>', methods=['POST'])
@login_required
def rename_audio(audio_id):
    new_name = request.form.get('new_name')
    if not new_name:
        flash("Filename cannot be empty.")
        return redirect(url_for('dashboard'))
        
    conn = get_db_connection()
    audio = conn.execute("SELECT * FROM audio_records WHERE id = ?", (audio_id,)).fetchone()
    if not audio:
        conn.close()
        flash('Audio file not found.')
        return redirect(url_for('dashboard'))
        
    if audio['user_id'] != current_user.id and current_user.role != 'admin':
        conn.close()
        flash('Permission denied.')
        return redirect(url_for('dashboard'))
        
    old_name = audio['original_filename']
    
    try:
        conn.execute("UPDATE audio_records SET original_filename = ? WHERE id = ?", (new_name, audio_id))
        conn.commit()
        log_action("Rename", f"Renamed audio from {old_name} to {new_name}")
        flash(f"Audio renamed to '{new_name}'.")
    except Exception as e:
        flash(f"Error renaming audio: {e}")
    finally:
        conn.close()
        
    return redirect(url_for('dashboard'))

@app.route('/audio/delete/<int:audio_id>', methods=['POST'])
@login_required
def delete_audio(audio_id):
    conn = get_db_connection()
    audio = conn.execute("SELECT * FROM audio_records WHERE id = ?", (audio_id,)).fetchone()
    if not audio:
        conn.close()
        flash('Audio file not found.')
        return redirect(url_for('dashboard'))
        
    if audio['user_id'] != current_user.id and current_user.role != 'admin':
        conn.close()
        flash('Permission denied.')
        return redirect(url_for('dashboard'))
        
    s3_key = audio['s3_object_key']
    filename = audio['original_filename']
    
    try:
        from utils import get_s3_client
        s3 = get_s3_client()
        if s3:
            s3.delete_object(Bucket=Config.S3_BUCKET_NAME, Key=s3_key)
            log_action("Delete", f"Deleted audio {filename} from S3")
            
        conn.execute("DELETE FROM audio_records WHERE id = ?", (audio_id,))
        conn.commit()
        flash(f"Audio '{filename}' deleted successfully.")
    except Exception as e:
        flash(f"Error deleting audio: {e}")
    finally:
        conn.close()
        
    return redirect(url_for('dashboard'))

@app.route('/audio/confirm_store/<int:audio_id>', methods=['POST'])
@login_required
def confirm_store_audio(audio_id):
    conn = get_db_connection()
    audio = conn.execute("SELECT * FROM audio_records WHERE id = ?", (audio_id,)).fetchone()
    if not audio:
        conn.close()
        flash('Audio file not found.')
        return redirect(url_for('dashboard'))
        
    if audio['user_id'] != current_user.id and current_user.role != 'admin':
        conn.close()
        flash('Permission denied.')
        return redirect(url_for('dashboard'))
        
    try:
        conn.execute("UPDATE audio_records SET status = 'completed' WHERE id = ?", (audio_id,))
        conn.commit()
        flash(f"Audio '{audio['original_filename']}' successfully stored.")
    except Exception as e:
        flash(f"Error storing audio: {e}")
    finally:
        conn.close()
        
    return redirect(url_for('dashboard'))

@app.route('/audio/status/<int:audio_id>')
@login_required
def audio_status(audio_id):
    conn = get_db_connection()
    audio = conn.execute("""
        SELECT status, similarity_score, matched_file, matched_transcript 
        FROM audio_records 
        WHERE id = ?
    """, (audio_id,)).fetchone()
    conn.close()
    
    if not audio:
        return jsonify({"status": "not_found"}), 404
        
    return jsonify({
        "status": audio['status'],
        "similarity": audio['similarity_score'],
        "matched_file": audio['matched_file'],
        "matched_transcript": audio['matched_transcript']
    })

@app.route('/video/status/<int:video_id>')
@login_required
def video_status(video_id):
    conn = get_db_connection()
    video = conn.execute("""
        SELECT status, similarity_score, matched_file, matched_transcript 
        FROM video_records 
        WHERE id = ?
    """, (video_id,)).fetchone()
    conn.close()
    
    if not video:
        return jsonify({"status": "not_found"}), 404
        
    return jsonify({
        "status": video['status'],
        "similarity": video['similarity_score'],
        "matched_file": video['matched_file'],
        "matched_transcript": video['matched_transcript']
    })

@app.route('/video/confirm_store/<int:video_id>', methods=['POST'])
@login_required
def confirm_store_video(video_id):
    conn = get_db_connection()
    video = conn.execute("SELECT * FROM video_records WHERE id = ?", (video_id,)).fetchone()
    if not video:
        conn.close()
        flash('Video file not found.')
        return redirect(url_for('dashboard'))
        
    if video['user_id'] != current_user.id and current_user.role != 'admin':
        conn.close()
        flash('Permission denied.')
        return redirect(url_for('dashboard'))
        
    try:
        conn.execute("UPDATE video_records SET status = 'completed' WHERE id = ?", (video_id,))
        conn.commit()
        flash(f"Video '{video['original_filename']}' successfully stored.")
    except Exception as e:
        flash(f"Error storing video: {e}")
    finally:
        conn.close()
        
    return redirect(url_for('dashboard'))

@app.route('/video/open/<int:video_id>')
@login_required
def open_video(video_id):
    conn = get_db_connection()
    video = conn.execute("SELECT * FROM video_records WHERE id = ?", (video_id,)).fetchone()
    conn.close()
    
    if not video:
        flash('Video file not found.')
        return redirect(url_for('dashboard'))
        
    if video['user_id'] != current_user.id and video['user_id'] is not None and current_user.role != 'admin':
        flash('Permission denied.')
        return redirect(url_for('dashboard'))
        
    s3_key = video['s3_object_key']
    filename = video['original_filename']
    
    from utils import get_s3_client
    s3 = get_s3_client()
    if not s3:
        flash("S3 service is not available.")
        return redirect(url_for('dashboard'))
        
    temp_path = os.path.join(Config.UPLOAD_TEMP, f"open_{s3_key}")
    try:
        s3.download_file(Config.S3_BUCKET_NAME, s3_key, temp_path)
        
        ext = filename.split('.')[-1].lower()
        mimetype = 'video/mp4'
        if ext == 'webm': mimetype = 'video/webm'
        elif ext == 'ogg': mimetype = 'video/ogg'
        elif ext == 'avi': mimetype = 'video/x-msvideo'
        elif ext == 'mov': mimetype = 'video/quicktime'
        
        def generate():
            with open(temp_path, 'rb') as f:
                yield from f
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        from flask import Response
        return Response(generate(), mimetype=mimetype,
                        headers={"Content-Disposition": f"inline;filename={filename}"})
    except Exception as e:
        flash(f"Error opening video: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return redirect(url_for('dashboard'))

@app.route('/video/rename/<int:video_id>', methods=['POST'])
@login_required
def rename_video(video_id):
    new_name = request.form.get('new_name')
    if not new_name:
        flash("Filename cannot be empty.")
        return redirect(url_for('dashboard'))
        
    conn = get_db_connection()
    video = conn.execute("SELECT * FROM video_records WHERE id = ?", (video_id,)).fetchone()
    if not video:
        conn.close()
        flash('Video file not found.')
        return redirect(url_for('dashboard'))
        
    if video['user_id'] != current_user.id and current_user.role != 'admin':
        conn.close()
        flash('Permission denied.')
        return redirect(url_for('dashboard'))
        
    old_name = video['original_filename']
    
    try:
        conn.execute("UPDATE video_records SET original_filename = ? WHERE id = ?", (new_name, video_id))
        conn.commit()
        log_action("Rename", f"Renamed video from {old_name} to {new_name}")
        flash(f"Video renamed to '{new_name}'.")
    except Exception as e:
        flash(f"Error renaming video: {e}")
    finally:
        conn.close()
        
    return redirect(url_for('dashboard'))

@app.route('/video/delete/<int:video_id>', methods=['POST'])
@login_required
def delete_video(video_id):
    conn = get_db_connection()
    video = conn.execute("SELECT * FROM video_records WHERE id = ?", (video_id,)).fetchone()
    if not video:
        conn.close()
        flash('Video file not found.')
        return redirect(url_for('dashboard'))
        
    if video['user_id'] != current_user.id and current_user.role != 'admin':
        conn.close()
        flash('Permission denied.')
        return redirect(url_for('dashboard'))
        
    s3_key = video['s3_object_key']
    filename = video['original_filename']
    
    try:
        from utils import get_s3_client
        s3 = get_s3_client()
        if s3:
            s3.delete_object(Bucket=Config.S3_BUCKET_NAME, Key=s3_key)
            log_action("Delete", f"Deleted video {filename} from S3")
            
        conn.execute("DELETE FROM video_records WHERE id = ?", (video_id,))
        conn.commit()
        flash(f"Video '{filename}' deleted successfully.")
    except Exception as e:
        flash(f"Error deleting video: {e}")
    finally:
        conn.close()
        
    return redirect(url_for('dashboard'))

@app.route('/alerts')
@login_required
def alerts():
    if current_user.role != 'admin':
        flash('Permission denied. Admin access required.')
        return redirect(url_for('dashboard'))
    
    include_dismissed = request.args.get('dismissed', 'false').lower() == 'true'
    all_alerts = suspicious_detector.get_all_alerts(include_dismissed=include_dismissed)
    
    return render_template('alerts.html', alerts=all_alerts, include_dismissed=include_dismissed)

@app.route('/alerts/<int:alert_id>/dismiss', methods=['POST'])
@login_required
def dismiss_alert(alert_id):
    if current_user.role != 'admin':
        flash('Permission denied.')
        return redirect(url_for('dashboard'))
    
    suspicious_detector.dismiss_alert(alert_id)
    flash('Alert dismissed successfully.')
    return redirect(url_for('alerts'))

@app.route('/api/activity-stats')
@login_required
def activity_stats():
    if current_user.role != 'admin':
        return jsonify({'error': 'Permission denied'}), 403
    
    user_id = request.args.get('user_id', type=int)
    hours = request.args.get('hours', default=24, type=int)
    
    if user_id:
        stats = suspicious_detector.get_user_stats(user_id, hours)
        return jsonify(stats)
    else:
        # Return overall stats
        alert_count = suspicious_detector.get_alert_count()
        return jsonify({'alert_count': alert_count})

if __name__ == '__main__':
    print("=" * 60)
    print("Starting Hybrid ML-CNS Deduplication System...")
    print("=" * 60)
    
    # Check if model exists, only train if needed
    print("\n[1/3] Checking ML Model...")
    if os.path.exists(Config.ML_MODEL_PATH):
        print("[OK] ML Model found, skipping training")
    else:
        print("  Training new ML Model...")
        try:
            ml_model.train(Config.ML_DATASET)
            print("[OK] ML Model trained successfully")
        except Exception as e:
            print(f"[X] ML Model training failed: {e}")
            print("  Continuing without ML predictions...")
    
    print("\n[2/3] Initializing database...")
    try:
        # Check if database exists and has tables
        db_needs_init = False
        
        # Check if tables exist
        try:
            from mysql_wrapper import get_mysql_connection
            conn = get_mysql_connection()
            cursor = conn.execute("SHOW TABLES LIKE 'users'")
            if cursor.fetchone() is None:
                print("  Database tables missing, initializing...")
                db_needs_init = True
            conn.close()
        except Exception:
            db_needs_init = True
        
        # Initialize database if needed
        if db_needs_init:
            from init_db import init_db
            init_db()
            print("[OK] Database initialized successfully")
        else:
            print("[OK] Database connection successful")
    except Exception as e:
        print(f"[X] Database error: {e}")
        print("  Please run: python init_db.py")
    
    print("\n[3/3] Starting Flask server...")
    print("=" * 60)
    print("[START] Server starting on http://127.0.0.1:5000")
    print("   Press CTRL+C to stop the server")
    print("=" * 60)
    
    try:
        app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)
    except OSError as e:
        if "address already in use" in str(e).lower():
            print("\n[X] ERROR: Port 5000 is already in use!")
            print("   Please stop other Python processes and try again.")
            print("   Run: Stop-Process -Name python -Force")
        else:
            print(f"\n[X] ERROR: {e}")
    except KeyboardInterrupt:
        print("\n\n[BYE] Server stopped by user")
