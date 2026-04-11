"""
WSGI config for medical_site project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/wsgi/
"""

import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'medical_site.settings')

# Выполняем миграции при запуске (только на Render)
try:
    if os.environ.get("RENDER"):
        from django.core.management import call_command
        call_command('migrate', interactive=False, verbosity=0)
        print("Migrations completed")
except Exception as e:
    print(f"Migration error: {e}")

application = get_wsgi_application()