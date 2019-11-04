RUBBER=rubber
SCREEN=screen
VIEWER=evince
PYTHON=python3

all: main.pdf

show: main.pdf
	$(SCREEN) -dm $(VIEWER) main.pdf

main.pdf: main.tex plots
	$(RUBBER) -d main.tex

plots: A2.py
	$(PYTHON) A2.py

clean:
	$(RUBBER) --clean main.tex
	rm *.pdf
	rm *.eps
