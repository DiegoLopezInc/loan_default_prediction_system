import os
import dotenv
import zipfile

dotenv.load_dotenv()
PROJECT_ROOT = os.getenv("PROJECT_ROOT")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "lending_club_data")


def unzip_lending_club_data():
