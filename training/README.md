# Training

You're now at the training phase. So far you have hand crafted some prompts and responses, and used `lab generate` to synthesize those prompt/response pairs into a new data set. Using a [Google Colab notebook](./Training_a_LoRA_With_Labrador.ipynb) and the NVidia T4 provided in the free tier, we will fine tune a LoRA. 

Pre-requisites: 
* [Google Colab](https://research.google.com/colaboratory/faq.html)
* A Gmail account that you're logged into, this will allow you to use Google Colab, which in the free tier will give you access to an NVidia T4 x 15GB GPU


**NOTE: At present, you'll need to download the notebook and upload it to Google Colab, once this repository is open sourced we will make an 'Open in Colab' button**

[The notebook](./Training_a_LoRA_With_Labrador.ipynb) in this folder will walk you through:
1. Uploading the output of `lab generate` (a synthetic dataset created based on your hand written prompts/responses).
2. Checking the base model before training
3. Setting up and training a Low Rank Adapter (LoRA). LoRA is a parameter efficient fine tuning method (PEFT) that allows you to fine tune a model on a small subset of the overall parameters, which allows you to conduct a finetuning in a fraction of the time, on a fraction of the hardware required. The resultant model should be updated and better handle your queries than the base model.
4. Inspecting the output model to make sure the LoRA training had the desired effect. (That is to say the the output has 'improved').
   
Once you have finished training and the output looks good, we encourage you  to open a [PR in taxonomy](https://github.com/open-labrador/taxonomy/pulls)!
