WHICH_CONTROLLER=$(lspci | grep "VGA compatible controller")
CONTAINERFILE="/default/Containerfile"
ifeq(grep NVIDIA <<< $WHICH_CONTROLLER, 0)
	CONTAINERFILE=/cuda/Containerfile
else ifeq(grep AMD <<< $WHICH_CONTROLLER, 0)
	Containerfile=/rocm/Containerfile
endif