"""
Runs preprocessing on real samples from train and test files to inspect before/after transformations.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import preprocess_record

sample_records = [
    # US record with legal suffix, apostrophe, and standard address
    {
        "entity_id": "S1-925783039",
        "business_name": "Orelee's Barbershop Inc.",
        "business_address": "1795 Westchester Drive, High Point, NC 27262",
        "country": "US"
    },
    # US record with unit number and abbreviation
    {
        "entity_id": "S1-106407869",
        "business_name": "Vision Partners Corp.",
        "business_address": "IA, Iowa City, 1064 Newton Rd, Unit 11",
        "country": "US"
    },
    # Indian record with Indic script in Telugu
    {
        "entity_id": "S2-620419239",
        "business_name": "ఇంటర్నేషనల్ సిస్టమ్స్ ప్రైవేట్ లిమిటెడ్",
        "business_address": "304, 4TH FLOOR, BHAGYA NAGAR, BALANAGAR, GANGADHARAPURAM, Telangana 500037",
        "country": "India"
    },
    # Indian record with Indic script in Devanagari (Hindi)
    {
        "entity_id": "S2-669353485",
        "business_name": "लक्ष्मी डेवलपर्स प्राइवेट लिमिटेड",
        "business_address": "H.NO #1338, NESARI, TAL-GADHINGLAJ, KOLHAPUR, KOLHAPUR, Maharashtra 416001",
        "country": "India"
    },
    # Indian record with complex plot/house number and abbreviations
    {
        "entity_id": "S1-109593962",
        "business_name": "M/s Great Impex Private Limited",
        "business_address": "6-2-101/5/C, Telangana, Hyderabad, Secunderabad, Lane Beside Centralview Apt New Bhoiguda 500003",
        "country": "India"
    },
    # French record with accents, legal suffix SASU, and French address
    {
        "entity_id": "S1-913506265",
        "business_name": "Thermal & Fils SASU",
        "business_address": "20 Rue Parmentier, Dunkerque, Hauts-de-France 59140",
        "country": "France"
    },
    # French record with leading noise prefix, SARL, and Boulevard
    {
        "entity_id": "S1-156285671",
        "business_name": "<< Team Ecole",
        "business_address": "175 Boulevard du Président Franklin Roosevelt, Bordeaux, Nouvelle-Aquitaine 33000",
        "country": "France"
    },
    # Record with completely missing/empty address
    {
        "entity_id": "S3-240268161",
        "business_name": "Urology Partners Industries",
        "business_address": None,
        "country": "US"
    }
]

print("=== REAL-WORLD SAMPLE TRANSFORMATIONS ===")
for rec in sample_records:
    p = preprocess_record(rec)
    print(f"\n[{p['country']}] ID: {p['entity_id']}")
    print(f"  Raw Name                  : '{p['business_name']}'")
    print(f"  -> name_normalized        : '{p['name_normalized']}'")
    print(f"  -> name_without_legal_suffix: '{p['name_without_legal_suffix']}'")
    print(f"  -> name_transliterated    : '{p['name_transliterated']}'")
    print(f"  -> name_compact           : '{p['name_compact']}'")
    print(f"  -> name_tokens            : {p['name_tokens']}")
    print(f"  Raw Address               : '{p['business_address']}'")
    print(f"  -> address_normalized     : '{p['address_normalized']}'")
    print(f"  -> house_number           : '{p['house_number']}'")
    print(f"  -> postal_code            : '{p['postal_code']}'")
    print(f"  -> numeric_tokens         : {p['numeric_tokens']}")
    print(f"  -> country                : '{p['country']}'")
