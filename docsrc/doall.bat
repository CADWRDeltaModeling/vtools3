call sphinx-apidoc --force -o . ../vtools
call make clean
call make html
call xcopy /Y /E /H _build\* ..\docs
