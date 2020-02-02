RUBBER=rubber
SCREEN=screen
VIEWER=xdg-open
PYTHON=python3
ZIP=zip

all: main.pdf

show: main.pdf
	$(SCREEN) -dm $(VIEWER) main.pdf

release: Sheet12GroupP.zip

Sheet12GroupP.zip: main.pdf *.py *.ipynb
	$(ZIP) $@  main.pdf *.py *.ipynb

main.pdf: main.tex
	$(RUBBER) -d main.tex

%.eps: %.svg
	inkscape $< -E $@ --export-ignore-filters --export-ps-level=3

clean: clean-temp
	rm -f *.pdf
	rm -f *.zip
	rm -f *.npy

clean-temp:
	$(RUBBER) --clean main.tex
	rm -f *.eps
	rm -f *-eps-converted-to.pdf
	rm -f *.log
