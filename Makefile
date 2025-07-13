.PHONY: run
run:
	python3 predict.py  ~/Videos/alprVideo.mp4 -w weights/wpodnet.pth --save-annotated images/annotated --save-warped images/warped --frame-skip 25
clean:
	rm  images/annotated/*.jpg
	rm  images/warped/*.jpg
	rm -rf outputs
