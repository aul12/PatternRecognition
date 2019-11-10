RUBBER=rubber
SCREEN=screen
VIEWER=evince
PYTHON=python3

all: main.pdf

show: main.pdf
	$(SCREEN) -dm $(VIEWER) main.pdf

main.pdf: main.tex
	$(RUBBER) -d main.tex

clean:
	$(RUBBER) --clean main.tex
	rm *.pdf
	rm *.eps
