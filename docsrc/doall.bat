call make clean
call sphinx-apidoc --force -o . ../vtools
call make html
call xcopy /Y /E /H _build\* ..\docs
rem This is needed because the clean will get rid of index.html
echo "<meta http-equiv="refresh" content="0; url=./html/index.html" />" > ..\docs\index.html