# README

You're now at the training phase. So far you have hand crafted some prompts and responses, and used `lab generate` to synthesize those prompt/response pairs into a new data set. 

The notebook in this folder will walk you through:
1. Uploading the output of `lab generate`
2. Checking the base model before training
3. Setting up and training a LoRA
4. Inspecting the output model to make sure the LoRA training had the desired effect. (That is to say the the output has 'improved').
   
There are some pre-requisites though- you will need a gmail account to run the notebook in Google Colab. We use Google Colab for this step because the free tier allows users to requisition an NVidia T4 x 15GB GPU. 

Once you have finished training out the output looks good, we encourage you to open a PR with your added taxonomy!
