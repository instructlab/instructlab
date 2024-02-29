# README

You're now at the training phase. So far you have hand crafted some prompts and responses, and used `lab generate` to synthesize those prompt/response pairs into a new data set. 

Pre-requisites: 
* A Gmail account that you're logged into, this will allow yout to use
* Google Colab, which in the free tier will give you access to an NVidia T4 x 15GB GPU
* [About Google Colab](https://research.google.com/colaboratory/faq.html)

**NOTE: At present, you'll need to download the notebook and upload it to Google Colab, once this repository is open sourced we will make an 'Open in Colab' button**

The notebook in this folder will walk you through:
1. Uploading the output of `lab generate`
2. Checking the base model before training
3. Setting up and training a LoRA
4. Inspecting the output model to make sure the LoRA training had the desired effect. (That is to say the the output has 'improved').
   
Once you have finished training out the output looks good, we encourage you to open a PR with your added taxonomy!
