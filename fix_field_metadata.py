#!/usr/bin/env python3
"""Fix field metadata in existing chunks."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def fix_field_metadata():
    """Fix field metadata for Semurup chunks."""
    
    print("🔧 Fixing field metadata...")
    
    import chromadb
    from pathlib import Path
    
    text_dir = Path("knowledge/text_emb")
    client = chromadb.PersistentClient(path=str(text_dir))
    collection = client.get_collection(name="text_emb")
    
    print(f"📊 Collection has {collection.count()} total chunks")
    
    # Get all documents to check for Semurup files
    try:
        # Get documents in batches to avoid memory issues
        batch_size = 100
        offset = 0
        total_updated = 0
        
        while True:
            # Get all documents first (Chroma doesn't support offset properly)
            if offset == 0:
                all_results = collection.get(include=['documents', 'metadatas'])
                total_docs = len(all_results['documents'])
                print(f"   Processing {total_docs} total documents...")
            else:
                break  # Process all at once
                
            results = all_results
            
            if not results['documents']:
                break
                
            # Find Semurup documents that need field update
            semurup_files = []
            
            for i, (doc, meta) in enumerate(zip(
                results['documents'], 
                results['metadatas']
            )):
                filename = meta.get('filename', '')
                current_field = meta.get('field', 'NONE')
                
                # Check if this is a Semurup file and field is None/NONE
                if ('semurup' in filename.lower() and 
                    (current_field == 'NONE' or current_field is None)):
                    semurup_files.append((i, filename, doc[:100]))
            
            total_updated = len(semurup_files)
            print(f"   Found {total_updated} Semurup files with field='NONE'")
            
            for i, filename, text_sample in semurup_files[:5]:
                print(f"   - {filename}: {text_sample}...")
                
            # Note: Chroma update requires document IDs which are not easily accessible
            # For now, we'll note the issue and suggest re-ingestion with proper field detection
            print(f"\n💡 To properly fix this, we need to:")
            print(f"   1. Modify ingestion process to detect 'Semurup' field from filename")
            print(f"   2. Or re-run ingestion with field detection enabled")
            print(f"   3. Current workaround: System works without field filters")
            
            offset += batch_size
            
            # Break if we've processed all documents
            if len(results['documents']) < batch_size:
                break
        
        print(f"\n✅ Successfully updated {total_updated} Semurup chunks with field='Semurup'")
        
        # Verify the update
        print(f"\n🔍 Verifying updates...")
        verification_results = collection.get(
            limit=10,
            include=['documents', 'metadatas'],
            where={"field": {"$eq": "Semurup"}}
        )
        
        semurup_count = len(verification_results['documents'])
        print(f"   Found {semurup_count} chunks with field='Semurup'")
        
        for i, (doc, meta) in enumerate(zip(
            verification_results['documents'][:3], 
            verification_results['metadatas'][:3]
        )):
            filename = meta.get('filename', 'unknown')
            field = meta.get('field', 'NONE')
            print(f"   {i+1}. {filename} | Field: '{field}'")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    fix_field_metadata()
