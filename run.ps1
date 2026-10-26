Write-Host "==============================================" -ForegroundColor Cyan
Write-Host " Starting Smart Placement Intelligence System" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

# 1. Check & create venv if not existing
if (!(Test-Path "venv\Scripts\Activate.ps1")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# 2. Activate venv
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
.\venv\Scripts\Activate.ps1

# 3. Install packages
Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Yellow
pip install -r requirements.txt

# 4. Open in browser
Write-Host "`nOpening http://127.0.0.1:5000 in your browser..." -ForegroundColor Green
Start-Process "http://127.0.0.1:5000"

# 5. Run Flask server
python app.py
