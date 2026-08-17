import os

# Doit etre defini AVANT le premier import de `api` (qui lit API_KEY au chargement
# du module), donc dans conftest.py : pytest le charge avant de collecter les tests.
os.environ.setdefault("API_KEY", "test-secret-key")
