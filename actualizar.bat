@echo off
REM Cambiar al directorio del proyecto
cd "c:\01 academico\001 Doctorado Economia UCAB\d tesis problema ahorro\5.4 experimentos\cuestionario\proyecto_encuesta"

REM Añadir todos los cambios al área de preparación
git add .

REM Hacer commit con un mensaje automático
git commit -m "Actualización automática"

REM Subir los cambios al repositorio remoto
git push origin main

echo Cambios subidos a GitHub correctamente.
pause
