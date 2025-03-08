bash reset.sh
ANONYMIZED_TELEMETRY=False \
venv/bin/python3 debug_kg_rag.py --output-dir ./out --db-path ./db --keep-temp --threads 16
#venv/bin/python3 test_kg_rag.py --input-file ./zto.txt --output-dir ./out --db-path ./db --keep-temp --threads 16
