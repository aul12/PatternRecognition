RUBBER=rubber
SCREEN=screen
VIEWER=xdg-open
PYTHON=python3
ZIP=zip

all: main.pdf

show: main.pdf
	$(SCREEN) -dm $(VIEWER) main.pdf

release: Sheet01GroupP.zip

Sheet01GroupP.zip: main.pdf A2.py
	$(ZIP) Sheet01GroupP.zip main.pdf A2.py

main.pdf: main.tex plots
	$(RUBBER) -d main.tex

plots: A2.py
	$(PYTHON) A2.py

clean:
	$(RUBBER) --clean main.tex
	rm *.pdf
	rm *.eps
	rm *.zip
