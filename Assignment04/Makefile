RUBBER=rubber
SCREEN=screen
VIEWER=xdg-open
PYTHON=python3
ZIP=zip

all: main.pdf

show: main.pdf
	$(SCREEN) -dm $(VIEWER) main.pdf

release: Sheet04GroupP.zip

Sheet04GroupP.zip: main.pdf
	$(ZIP) $@  main.pdf *.py *.ipynb

main.pdf: main.tex A1.eps
	$(RUBBER) -d main.tex

A1.eps: A1.py
	$(PYTHON) A1.py

clean: clean-temp
	rm -f *.pdf
	rm -f *.zip
	rm -f *.npy

clean-temp:
	$(RUBBER) --clean main.tex
	rm -f *.eps
	rm -f *-eps-converted-to.pdf
	rm -f *.log
