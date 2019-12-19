# -*- makefile -*-
.PHONY: main clean test pip linder

# OS-specific configurations
ifeq ($(OS),Windows_NT)
	PYTHON_exe = python.exe

else
	UNAME_S := $(shell uname -s)


	ifeq ($(UNAME_S),Linux) # Linux
		PYTHON_exe=python

	endif

	ifeq ($(UNAME_S),Darwin) # macOS
		PYTHON_exe=python

	endif

endif



PYTHON := $(if $(PYTHON_exe),$(PYTHON_exe),python)
# All the files which include modules used by other modules (these therefore
# need to be compiled first)

MODULE = linder

main:
	$(MAKE) clean
	${PYTHON} setup.py bdist_wheel

# make fortran exe and run test cases
test:
	${PYTHON} setup.py test


# If wanted, clean all *.o files after build
clean:
	rm -rf ${MODULE}/*.so ${MODULE}/*.pyc ${MODULE}/__pycache__ ${MODULE}/*.dSYM
	rm -rf ${MODULE}/test/*.pyc ${MODULE}/test/__pycache__
	rm -rf ${MODULE}/util/*.pyc ${MODULE}/util/__pycache__
	rm -rf ${MODULE}/cmd/*.pyc ${MODULE}/cmd/__pycache__
	rm -rf build dist

# upload wheels to pypi using twine
upload:
	twine upload --skip-existing dist/*whl
