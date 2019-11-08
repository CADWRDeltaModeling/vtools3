rem only needed if you add submodules etc..
rem sphinx-apidoc -o . ../vtools
make clean && make html
xcopy /Y /E /H _build\* ..\docs
