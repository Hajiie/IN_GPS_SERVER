import os
from django.core.management import call_command
from waitress import serve

# --- Django Project Setup ---
# Set the DJANGO_SETTINGS_MODULE environment variable to point to your project's settings.
# Replace 'IN_GPS_SERVER' with your actual project name if it's different.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'IN_GPS_SERVER.settings')

import django
# Load Django settings and set up the application environment.
django.setup()

# Import the WSGI application object after Django setup.
from IN_GPS_SERVER.wsgi import application

def run():
    """
    This function performs initial database setup and then starts the web server.
    """
    # --- 1. Automatic Database Migration ---
    # This is the core logic for automatic database creation.
    # When this script runs for the first time, it will create the .sqlite3 file
    # and apply all necessary table structures based on your models.
    # The `interactive=False` option prevents any user prompts during migration.
    print("Starting database initialization...")
    call_command('migrate', interactive=False)
    print("Database setup complete.")
    
    # --- 2. Start the Production Web Server ---
    # After migrations are successfully applied, start the server using waitress.
    host = '127.0.0.1'
    port = 8000
    print(f"Starting server at http://{host}:{port}")
    serve(application, host=host, port=port)

if __name__ == "__main__":
    run()
