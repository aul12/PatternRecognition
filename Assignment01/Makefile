RUBBER=rubber
SCREEN=screen
VIEWER=evince
PYTHON=python3

all: main.pdf

show: main.pdf
	$(SCREEN) -dm $(VIEWER) main.pdf

main.pdf: main.tex plots
	$(RUBBER) -d main.tex

plots: A24.py
	$(PYTHON) A24.py

clean:
	$(RUBBER) --clean main.tex
	rm *.pdf
	rm *.eps
