RUBBER=rubber
SCREEN=screen
VIEWER=evince

all: main.pdf

show: main.pdf
	$(SCREEN) -dm $(VIEWER) main.pdf

main.pdf: main.tex
	$(RUBBER) -d main.tex

clean:
	$(RUBBER) --clean main.tex
	rm *.pdf
	rm *.eps
