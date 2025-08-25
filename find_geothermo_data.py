#!/usr/bin/env python3
"""Find geothermometer data in Semurup files."""

import fitz
import os
import re

def find_temperature_data():
    """Search for temperature and geothermometer data."""
    
    files = [
        '3D MT and Gravity Modeling - Semurup.pdf',
        'Geokimia - Laporan Akhir Survey Geokimia Semurup 2022_Final.pdf', 
        'PRE FS SEMURUP (2014).pdf',
        'Laporan Akhir Survey Geologi Daerah Semurup 2022.pdf'
    ]
    
    base_dir = 'knowledge/Spesific/'
    
    for filename in files:
        try:
            filepath = os.path.join(base_dir, filename)
            if not os.path.exists(filepath):
                continue
                
            print(f"\n🔍 === {filename} ===")
            doc = fitz.open(filepath)
            
            pages_with_temp = []
            geothermo_pages = []
            
            for i in range(min(30, len(doc))):  # Check first 30 pages
                text = doc[i].get_text().lower()
                
                # Check for geothermometer
                if any(term in text for term in ['geotherm', 'geotermometer']):
                    geothermo_pages.append((i+1, doc[i].get_text()[:300]))
                
                # Check for temperature data
                temp_matches = re.findall(r'(\d+\.?\d*)\s*°?c', text)
                if temp_matches:
                    pages_with_temp.append((i+1, temp_matches))
            
            print(f"📊 Found {len(geothermo_pages)} pages with geothermometer")
            print(f"🌡️ Found {len(pages_with_temp)} pages with temperature")
            
            # Show geothermometer pages
            for page_num, text in geothermo_pages[:2]:
                print(f"\n📋 Page {page_num} (Geothermometer):")
                print(text[:400] + "...")
                
            # Show temperature pages
            for page_num, temps in pages_with_temp[:3]:
                if any(float(t) > 50 for t in temps if t.replace('.','').isdigit()):
                    print(f"\n🌡️ Page {page_num} (High Temperatures): {temps}")
                    page_text = doc[page_num-1].get_text()
                    # Find context around temperature
                    for temp in temps:
                        if temp.replace('.','').isdigit() and float(temp) > 50:
                            temp_pos = page_text.lower().find(f"{temp}°c")
                            if temp_pos == -1:
                                temp_pos = page_text.lower().find(f"{temp} °c")
                            if temp_pos != -1:
                                context_start = max(0, temp_pos - 100)
                                context_end = min(len(page_text), temp_pos + 200)
                                context = page_text[context_start:context_end]
                                print(f"   Context: ...{context}...")
                                break
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

if __name__ == "__main__":
    find_temperature_data()
