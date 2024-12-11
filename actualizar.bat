cd "c:/01 academico/001 Doctorado Economia UCAB/d tesis problema ahorro/5.4 experimentos/cuestionario/proyecto_encuesta/"
git add .
git commit -m "Actualización de código"
git push origin main

IF %ERRORLEVEL% EQU 0 (
    echo "Cambios subidos exitosamente a GitHub"
) ELSE (
    echo "No hay cambios nuevos para subir o ocurrió un error"
)
pause

