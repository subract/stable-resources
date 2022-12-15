A tutorial for training a Stable Diffusion model with Dreambooth (software used to teach Stable Diffusion models new concepts) and RunPod.io (an inexpensive, GPU-focused VPS provider) - for under $1 per model. I’m basically summarizing and expanding upon the 42 minute tutorial from [Aitrepreneur](https://www.youtube.com/watch?v=7m__xadX0z0), so use that as a reference if this guide is unclear.

# Here be dragons

Understand what you’re creating: an AI model trained on your likeness. It will be capable of generating just about anything and putting your face into it. It can be used to create wildly cool artwork, but just as easily create incriminating or NSFW artwork of yourself. These instructions will walk you through generating the model on a remote server, then downloading it to your computer and deleting the remote copy. I recommend you keep control of the result. It should go without saying that you should never train a model on anther person without their consent.

If we’re so concerned about keeping control of model, why not generate it locally on our own machine? While possible, model training requires a powerful GPU (at least right now) with at least 24GB of VRAM - that means an $1,000 MSRP RTX 3090 is about the entry point. Most people, like me, can’t run the training locally, so renting one is necessary. Fortunately, once the model is trained, actually generating images can be done on more humble GPUs (you’ll still need at least 4GB of VRAM or so).

# Prerequisites

## Select training images

We’ll focus on training a custom person into the model. You’ll want to use 10-20 training images (use an even number of images), all cropped and resized to 512 x 512px. You can use the browser-based [Bulk Image Resizing Made Easy](https://www.birme.net/) to make the process easier - just upload, set it to 512x512, drag the squares to the appropriate crop, and redownload the results. Shoot for about 2-3 full body, 3-5 upper body, and 5-12 close portraits. Since we're resizing them to be so small, don't stress about using a nice camera.

The images you use will impact your final model! Dreambooth is going to learn that whatever is in common between the photos **is** the subject you intend to train it upon. Wearing the same shirt in every picture? Dreambooth is going to think that a green shirt is part of your essence. Every photo taken in a white-walled bedroom? Don't expect Dreambooth to figure out that you can be in other rooms as well. The more diversity in your photo set - different angles, lighting, outfits/accessories, backgrounds - the better idea Dreambooth will have about what makes you unique.

## Set up Stable Diffusion locally

While not strictly necessary to train the model, you’ll need a way to actually run the model once you’ve trained it. There are a number of user interfaces out there for generating images. The one I find the simplest (for Windows users) is [NMKD’s](https://nmkd.itch.io/t2i-gui), an open-source GUI [(GitHub)](https://github.com/n00mkrad/text2image-gui) that’s easy to set up and use. However, NMKD does **not** support safety filters to try to detect and filter explicit images. If that’s a requirement, consider [cmdr2’s](https://github.com/cmdr2/stable-diffusion-ui) UI (I haven’t tried it myself, but it has a toggle for the NSFW filter). In my experience, Stable Diffusion will *rarely* generate NSFW content unprompted, but not *never*.

# Train the model

We'll use [RunPo.io](https://runpod.io) to train our models. Create an account and fund it (you must add at least $10, but the credit will stick around on your account until you use it). There are other free options out there, most notably [Google Colab](https://colab.research.google.com/), but - call me paranoid - I don’t personally relish the idea of giving Google an AI model of my likeness.

For our training tool, we’ll use [the JoePenna Dreambooth repository](https://github.com/JoePenna/Dreambooth-Stable-Diffusion). It’s a version of Dreambooth optimized for training on people, among other things. It’s written as a [Jupyter notebook](https://jupyter.org/) - basically an interactive document with Python commands inside of “cells” that can be executed independently.

1. Provision a new pod with at least 24GB VRAM in the Secure Cloud environment (The Community cloud is marginally cheaper, but runs on community servers instead of centrally-hosted one)
	1. Use the `RunPod Pytorch` template
	2. Leave the disk sizes at default values
	3. Use encrypted storage
	4. Be sure to deploy “On-Demand” instead of “Spot” - otherwise, another user could kick you off in the middle of training!
3. Wait a minute or two for it to provision, then click Connect > Connect to Jupyter Lab
4. Under Other, click Terminal to open a shell
5. Run the following:

```bash
git clone https://github.com/JoePenna/Dreambooth-Stable-Diffusion.git
```

7. Once complete, use the directory pane on the left to navigate into the `Dreambooth-Stable-Diffusion` directory open `dreambooth_runpod_joepenna.ipynb`
5. Run the `BUILD ENV` cell to install packages (click inside the cell to select it, then click the Run button in the top bar). Wait for it to complete before continuing.
6. Click back over to your terminal and run the following to download Stable Diffusion v1.5 from HuggingFace (a repository of AI models and code)
	- Note that you can use other models as a base as well - for example [Arcane Diffusion](https://huggingface.co/nitrosocke/Arcane-Diffusion) or [bubblydubbly](https://huggingface.co/Marre-Barre/bubblydubbly/tree/main). Just replace the download link below with a link to the `.ckpt` file for the model. I’d recommend getting comfortable with the basic process before trying this.

```bash
cd Dreambooth-Stable-Diffusion
wget -O model.ckpt https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt
```

7. Back in the notebook, scroll down a bit and run the `Download Regularization Images` cell (right-click > Clear Outputs to remove the obnoxiously large error afterwards - you can safely ignore it).
	- This will download a reference set of images of a person to use as a base.
8. Make a new directory called `training_images` and upload your training images to it by dragging and dropping
	- Make sure you're using an even number of images
	- Make sure you’re using the images you resized to 512x512
9. Edit the parameters in the `Training` cell to appropriate values:
	- `project_name`: whatever you’d like
	- `max_training_steps`: 100x the number of images you uploaded, e.g. 1800 for 18 images
	- `token`: this will be how you’ll refer to the person - you could use your name, a random string, or really anything. Just pick something that won’t be an existing token in the model already
10. We’re finally ready to train! Run the `Training` cell. Wait for training to complete - it’ll take a while. For my example of 1800 steps, an RTX 3090 took around 45 minutes.
	- Every 500 steps by default, a pair of sample images will be generated - one with the subject, one without. You can find them in the `logs/{project_name}/images/train/` directory. Use these to get an idea of how your model is coming along!
11. Run the `Copy and name the checkpoint file` cell
12. Download the new model from the `trained_models` directory. I found that the bandwidth is poor, so expect the ~2GB file to download slowly.
13. Switch back over to the RunPod interface and stop the pod from `My Pods` on the left pane, then click the trash can to terminate it
	- If you stop the pod, but don’t terminate it, your model will be available to download again in the future, but you’ll continue to be charged a nominal fee for storage. I recommend terminating pods you’re done with, and re-running these same steps in the future, unless you’re generating multiple models back-to-back.
	- Once you’ve terminated the pod, RunPod *should* no longer have a copy of your model

## Helpful tips

- If your Jupyter notebook cells get stuck in an odd state or don’t seem to run correctly, restart the Python kernel with the restart icon on the top bar
- You should be able to use a model you generated as the starting point for another one by copying it to the root of the git repo as `model.ckpt`, replacing the model we downloaded initially. Be sure to replace the training images with a new set and choose a new `token`. This will allow you to train multiple people into a single model.
- At the top of RunPod’s site, you can see your remaining credit with a helpful indicator letting you know how quickly it’s being depleted by your currently running resources

# Using the model

Load the model into the UI of your choice (see [[#Set up Stable Diffusion locally]] above). If you’re using NMKD’s GUI, drop your downloaded model into the `Data\models`, then select it from the settings menu.

To reference yourself, you must use use `<token> person`, not just `<token>`! For example, if you set your token to `johndoe`, you could use the following prompt:

```
johndoe person as a paladin warrior, ultra detailed fantasy, dndbeyond, bright, colourful, realistic, dnd character portrait, pathfinder, pinterest, art by ralph horsley, dnd, rpg, lotr game design fanart by concept art, behance hd, artstation, deviantart, hdr render in unreal engine 5
```

Crafting prompts like this one for Stable Diffusion is called ”prompt engineering”, and is an art form unto itself. Sites like [PromptHero]([https://prompthero.com](https://prompthero.com/)) can help you find a starting point, but - as ever - be aware of NSFW content. OpenAI publishes a [free guidebook](https://openart.ai/promptbook) on how to craft prompts.

Happy diffusing!
