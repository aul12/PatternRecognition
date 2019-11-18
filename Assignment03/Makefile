RUBBER=rubber
SCREEN=screen
VIEWER=xdg-open
PYTHON=python3
ZIP=zip

all: main.pdf

show: main.pdf
	$(SCREEN) -dm $(VIEWER) main.pdf

release: Sheet03GroupP.zip

Sheet03GroupP.zip: main.pdf
	$(ZIP) Sheet03GroupP.zip main.pdf *.py

main.pdf: main.tex
	$(RUBBER) -d main.tex

clean: clean-temp
	rm -f *.pdf
	rm -f *.zip
	rm -f *.npy

clean-temp:
	$(RUBBER) --clean main.tex
	rm -f *.eps
	rm -f *-eps-converted-to.pdf
	rm -f *.log
