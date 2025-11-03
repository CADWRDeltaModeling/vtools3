call make clean
call sphinx-apidoc --force -o . ../vtools
call make html
call xcopy /Y /E /H _build\* ..\docs
