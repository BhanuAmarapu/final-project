import json
import numpy as np
from mysql_wrapper import get_mysql_connection
from utils import log_action

def cosine_similarity(v1, v2):
    """Calculate the cosine similarity between two numeric vectors."""
    a = np.array(v1)
    b = np.array(v2)
    if a.size == 0 or b.size == 0:
        return 0.0
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))

class SimilarityService:
    def __init__(self):
        pass

    def find_highest_similarity(self, new_embedding, exclude_id=None, table_name="audio_records"):
        """
        Compare new embedding against all stored transcript embeddings in target table.
        Returns:
            dict: {
                "similarity": float (0.0 to 1.0),
                "matched_record": dict (database row fields) or None
            }
        """
        log_action("Similarity Calculation Started", f"Comparing uploaded embedding against {table_name} records.")
        
        if not new_embedding:
            return {"similarity": 0.0, "matched_record": None}
        
        # Validate table name to prevent SQL injection
        if table_name not in ["audio_records", "video_records"]:
            raise ValueError("Invalid target similarity table name")
  
        # Connect to DB and fetch all stored records
        conn = get_mysql_connection()
        try:
            if exclude_id is not None:
                cursor = conn.execute(
                    f"SELECT id, original_filename, transcript, embedding, language, duration, s3_object_key FROM {table_name} WHERE id != ?",
                    (exclude_id,)
                )
            else:
                cursor = conn.execute(
                    f"SELECT id, original_filename, transcript, embedding, language, duration, s3_object_key FROM {table_name}"
                )
            stored_records = cursor.fetchall()
        except Exception as e:
            print(f"[SimilarityService] DB Error: {e}")
            stored_records = []
        finally:
            conn.close()
 
        if not stored_records:
            log_action("Highest Similarity Found", f"No existing {table_name} records to compare against.")
            return {"similarity": 0.0, "matched_record": None}

        highest_similarity = 0.0
        best_match = None

        for record in stored_records:
            try:
                # The embedding is stored as a JSON string, deserialize it
                stored_emb_str = record['embedding']
                if not stored_emb_str:
                    continue
                stored_emb = json.loads(stored_emb_str)
                
                similarity = cosine_similarity(new_embedding, stored_emb)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = record
            except Exception as e:
                rec_id = record['id'] if ('id' in record or hasattr(record, '__getitem__')) else 'unknown'
                print(f"[SimilarityService] Error comparing record ID {rec_id}: {e}")
                continue

        log_action("Highest Similarity Found", f"Score: {highest_similarity * 100:.2f}% | File: {best_match['original_filename'] if best_match else 'None'}")
        
        return {
            "similarity": highest_similarity,
            "matched_record": best_match
        }

similarity_service = SimilarityService()
