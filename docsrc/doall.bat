sphinx-apidoc --force -o . ../vtools
make clean && make html
xcopy /Y /E /H _build\* ..\docs
