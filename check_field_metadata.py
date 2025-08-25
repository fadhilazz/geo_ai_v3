#!/usr/bin/env python3
"""Check field metadata in chunks."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def check_field_metadata():
    """Check field metadata in knowledge base."""
    
    print("🔍 Checking field metadata...")
    
    import chromadb
    from pathlib import Path
    
    text_dir = Path("knowledge/text_emb")
    client = chromadb.PersistentClient(path=str(text_dir))
    collection = client.get_collection(name="text_emb")
    
    # Get sample documents
    print(f"\n📊 Collection has {collection.count()} total chunks")
    
    # Look for Semurup files specifically
    try:
        results = collection.get(
            limit=20, 
            include=['documents', 'metadatas']
        )
        
        semurup_docs = []
        field_values = set()
        
        for i, (doc, meta) in enumerate(zip(results['documents'], results['metadatas'])):
            filename = meta.get('filename', '')
            field = meta.get('field', 'NONE')
            field_values.add(field)
            
            if 'semurup' in filename.lower() or 'semurup' in doc.lower():
                semurup_docs.append((filename, field, meta.get('page', 0), doc[:100]))
                
        print(f"\n📋 Field values found: {sorted(field_values)}")
        print(f"🎯 Semurup documents found: {len(semurup_docs)}")
        
        for i, (filename, field, page, text) in enumerate(semurup_docs[:5]):
            print(f"\n{i+1}. File: {filename}")
            print(f"   Field: '{field}'")
            print(f"   Page: {page}")
            print(f"   Text: {text}...")
            
        # Check why field is None
        if semurup_docs and all(field == 'None' or field is None for _, field, _, _ in semurup_docs):
            print(f"\n❌ PROBLEM: All Semurup documents have field='None'")
            print(f"💡 This suggests field detection during ingestion didn't work")
            print(f"🔧 Need to fix field detection in classification process")
        else:
            print(f"\n✅ Field detection working properly")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_field_metadata()
