@echo off
cd /d D:\Pycharm\pose

if exist .git (
    echo [1/5] Remove old .git ...
    rmdir /s /q .git
)

echo [2/5] Init git repo ...
git init -b main
git remote add origin https://github.com/waive66/pose.git

echo [3/5] Stage files ...
git add .

echo [4/5] Commit ...
git commit -m "Init pose project"

echo [5/5] Push to GitHub ...
git push -u origin main

echo.
echo Done!
pause
