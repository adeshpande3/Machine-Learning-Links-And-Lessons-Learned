# Machine Learning Links and Lessons Learned

List of all the lessons learned, best practices, and links from my time studying machine learning. 

* [Learning Machine Learning](#learning-machine-learning)
* [Best Courses](#best-courses)
* [Most Important Deep Learning Papers](#most-important-deep-learning-papers)
* [Cool Use Cases of ML](#cool-use-cases-of-ml)
* [ML Tech Talks](#ml-tech-talks)
* [Best Blogs](#best-blogs)
* [Data and Features](#data-and-features)
* [Models](#models)
* [Hyperparameters](#hyperparameters)
* [Tensorflow/Deep Nets](#tensorflowdeep-nets)
* [Deep Learning Frameworks](#deep-learning-frameworks)
* [CNNs](#cnns)
* [NLP](#nlp)
* [Deep Reinforcement Learning](#deep-reinforcement-learning)
* [ML Project Advice](#ml-project-advice)
* [Math Resources](#math-resources)
* [Bias in Machine Learning](#bias-in-machine-learning)
* [Kaggle](#kaggle)
* [Debugging ML Models](#debugging-ml-models)
* [Best Python Libraries for ML](#best-python-libraries-for-ml)
* [Other Interesting Links](#other-interesting-links)
* [UCLA ACM AI Resources](#ucla-acm-ai-resources)
* [Random Thoughts](#random-thoughts)
* [Research Ideas](#research-ideas)
* [Other](#other)

## Learning Machine Learning

"How do you get started with machine learning?". With AI and ML becoming such huge words in the tech industry, it's hard to go a full week without hearing something along these lines on online forums, in discussions with other students at UCLA, and even from fellow pre-meds and humanities majors. From my own experience of getting familiar with ML and from my experiences of teaching others through ACM AI, here's my best response to that question. 

1) Before getting started with any code or any technical terms, I think the best first step is to gain a big picture understanding of what machine learning is, and what it is attempting to do. When first teaching other students about ML, I've found that it's **very** important to give them a general understanding of the field before starting to dive into terms like gradient descent and loss function. Machine learning, as stated by Wikipedia, is a *field of computer science that gives computers the ability to learn without being explicitly programmed*. I might also add that machine learning is a subfield of AI, and that it is a unique approach to creating intelligent systems by making use of training data and optimization. I'd recommend the following high level links and videos to get you comfortable with the field as a whole. 
	* [Machine Learning Introduction](https://www.youtube.com/watch?v=seG9J49bBYI): Loved this video because it starts with great definitions and introduces you to important terminology. Feel free to stop at 6:47.
	* [What is Machine Learning?](https://www.youtube.com/watch?v=WXHM_i-fgGo): Explains the 3 different subareas of machine learning: supervised learning, unsupervised learning, reinforcement learning
	*  [A Friendly Intro to Machine Learning](https://www.youtube.com/watch?v=IpGxLWOIZy4): Great video with some cool illustrations, but honestly I think just watching until 5:54 is sufficient. 
	* [Basic Machine Learning Algorithms Overview](https://www.youtube.com/watch?v=ggIk08PNcBo): Don't worry about knowing exactly what every one of these terms mean. Just get a sense for the different algorithms and the tasks they are used for. We'll go into way more detail later on. 
	* [Machine Learning from Zero to Hero](https://medium.freecodecamp.org/machine-learning-how-to-go-from-zero-to-hero-40e26f8aa6da): Get motivated to learn Machine Learning.  Sometimes a simple sitting back and watching the right videos can ignite and fan the fire to get active in ML.  If you know software, this is a great starting post. Don't worry about knowing every single detail in each of the videos, but rather think about the high level goal of ML. 

2) Okay cool, so now you should have a general idea of the goal of machine learning. We want to be able to create a system that is able to perform some task and we want to train this system using a dataset that we have. Let's now jump into some of the models used in machine learning. An approach that has helped me is **learning about these models one at a time**, while also constantly thinking about the similarities and differences between them. For each model, I think going through the following process would be helpful:
	* Understand the high level approach that the model is taking. What type of task is the model trying to solve, and how is it going about solving it?
	* Start to go into the specifics of what goes into the model. Think about what function the model is trying to compute. What is the loss function that is commonly used? Does this model use gradient descent for its optimization procedure? 
	* Think about an example where this model could is used. Is the model used for classification tasks or for regression tasks? In the particular example that you thought of, what do the inputs and outputs to the model look like? What type of dataset would you need to train this algorithm?
	* Transition to more practical exercises. If you're comfortable with coding, try to reconstruct the model in code. Think about how you would code the function, compute the loss, the gradients, etc. If that's a bit too intimidating right now (don't worry, it was initially for me too!), then just try to write down some pseudo-code and then look online to see how other people did it!
	* Finally, I think the coolest part of learning any machine learning algorithm is getting to see it in practice. The final step would be to do some sort of project or experiment where you run either the algorithm you coded up in the previous step or one imported from SciKitLearn on a particular problem of your choice. 
	* To make sure that you've really understood the model, it's helpful to just quiz yourself. Take a piece of paper, and (from memory) write down a summary of the model as if you were explaining it to a 5 year old. Then, write a technical summary that talks about what function is being computed, how the training procedure occurs, what the loss function is, etc. See if you can explain it to a friend. Think about a new problem space this model could be used for. Before jumping into the next model to learn, really ask yourself if you've firmly understood the concepts (and be honest with yourself here!) and then proceed accordingly. If the answer is still no, YouTube and Google are your friends :) With the scale of content that we have with this field, it won't be hard to find a video or tutorial that has the answers to your questions.  

	Okay, that was definitely a lot of info. The tl;dr is that learning about machine learning models has a lot of steps involved. First, there's understanding the high level perspective, then there's identifying the unique characteristics of the model, then it's testing yourself to see if you can code it up, and finally it's using the model in a practical setting. Don't worry if this takes you a week or even a month with something like linear regression. Going through this process slowly and making sure that you're understanding each step is critical for retaining the information. 

For steps #3 - 7, repeat the process I talked about above with each of the following ML models. I have listed a couple of links per model. You don't have to watch of all them per se, just wanted to include as many good resources I could find. **If you need more material or you still don't understand a topic, do a simple YouTube or Google search!** You'll be surprised with how much you find.

3) **Linear Regression**: 
	* [Coursera Course - Linear Regression with One Variable](https://www.youtube.com/watch?v=kHwlB_j7Hkc)
	* [Simple Linear Regression](https://www.youtube.com/watch?v=ZkjP5RJLQF4)
	* [Linear Regression for Machine Learning](https://machinelearningmastery.com/linear-regression-for-machine-learning/)

4) **Logistic Regression**: 
	* [Coursera Course - Logistic Regression and Classification](https://www.youtube.com/watch?v=-la3q9d7AKQ)
	* [Logistic Regression - An Introduction](https://www.youtube.com/watch?v=zAULhNrnuL4)
	* [Stanford Logistic Regression Overview](http://ufldl.stanford.edu/tutorial/supervised/LogisticRegression/)
	* [Siraj Raval Logistic Regression Tutorial](https://www.youtube.com/watch?v=D8alok2P468)

5) **K Nearest Neighbors**: 
	* [K Nearest Neighbors](https://www.youtube.com/watch?v=k_7gMp5wh5A)
	* [GeeksForGeeks Explanation](https://www.geeksforgeeks.org/k-nearest-neighbours/)
	* [Udacity Explanation of KNNs](https://www.youtube.com/watch?v=mpU84OJ5vdQ)

6) **K-Means**: 
	* [Coursera Course - K Means](https://www.youtube.com/watch?v=hDmNF9JG3lo)
	* [Stanford K Means Overview](http://stanford.edu/~cpiech/cs221/handouts/kmeans.html)
	* [Siraj Raval K Means Tutorial](https://www.youtube.com/watch?v=9991JlKnFmk)

7) **Decision Trees**:
	* [Nando de Freitas Lecture](https://www.youtube.com/watch?v=-dCtJjlEEgM)
	* [Decision Trees in ML](https://towardsdatascience.com/decision-trees-in-machine-learning-641b9c4e8052)

**Optional, but Worth Your Time**: Random Forest, SVMs, Naive Bayes, Gradient Boosted Methods, PCA

So now that we have a decent understanding of some ML models, I think that we can transition into deep learning. 

8) **Neural Networks**: If someone wants to get started with deep learning, I think that the best approach is to first get familiar with machine learning (which you all will have done by this point) and then start with neural networks. Following the same high level understanding -> model specifics -> code -> practical example approach would be great here as well. 
	* [3Blue1Brown Neural Network Playlist](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi): Can't overstate how well made these videos are. They go in depth on backpropagation and gradient descent, two critical concepts for understanding neural networks. 
	* [How Deep Neural Networks Work](https://www.youtube.com/watch?v=ILsA4nyG7I0): Another great tutorial by Brandon Rohrer.
	* [A Friendly Introduction to Deep Learning and Neural Networks](https://www.youtube.com/watch?v=BR9h47Jtqyw): Another visually appearing presentation of neural nets.
	* [Neural Network Playground](http://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.09661&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false): Great web app, created by Google, that allows you to tinker with a neural network in the browser. Awesome for gaining some practical understanding. 
	* [Michael Nielsen Book on NNs](http://neuralnetworksanddeeplearning.com/chap1.html): Very in depth and comprehensive.

9) **Convolutional Neural Networks**:  A convolutional neural network is a special type of neural network that has been successfully used for image processing tasks.
	* [A Beginner's Guide to Understanding CNNs](https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/): Shameless plug LOL
	* [CS 231N Homepage](http://cs231n.github.io/convolutional-networks/): Stanford CS231N is a grad course focused on CNNs that was originally taught by Fei Fei Li, Andrej Karpathy, and others.
	* [CS 231N Video Lectures](https://www.youtube.com/playlist?list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC): All the lecture videos from 2016. There will likely be a playlist for 2017 somewhere on YouTube as well. 
	* [Brandon Rohrer YouTube Tutorial](https://www.youtube.com/watch?v=FmpDIaiMIeA): Great visuals on this tutorial video. 
	* [Andrew Ng's CNN Course](https://www.youtube.com/playlist?list=PLBAGcD3siRDjBU8sKRk0zX9pMz9qeVxud): Videos from Andrew Ng's deep learning course. 

10) **Recurrent Neural Networks**: A recurrent neural network is a special type of neural network that has been successfully used for natural language processing tasks.
	* [Deep Learning Research Paper Review: NLP](https://adeshpande3.github.io/adeshpande3.github.io/Deep-Learning-Research-Review-Week-3-Natural-Language-Processing): Too many shameless plugs or nah? LOL
	* [CS 224D Video Lectures](https://www.youtube.com/playlist?list=PLCJlDcMjVoEdtem5GaohTC1o9HTTFtK7_): Stanford CS 224D is a grad course focused on RNNs and applying deep learning to NLP. 
	* [RNNs and LSTMs](https://www.youtube.com/watch?v=WCUNPb-5EYI): We all love Brandon honestly. 
	* [Recurrent Neural Networks - Intel Nervana](https://www.youtube.com/watch?v=Ukgii7Yd_cU): Very comprehensive.
	* [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/): Chris Olah's posts are readable, yet in-depth.
	* [Introduction to RNNs](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/): Denny Britz is another great author who has a wide ranging blog.

11) **Reinforcement Learning**: While the 3 prior ML methods are necessarily important for understanding RL, a lot of recent progress in this field has combined elements from the deep learning camp as well as from the traditional reinforcement learning field. 
	* [David Silver's Reinforcement Learning Course](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PL5X3mDkKaJrL42i_jhE4N-p6E2Ol62Ofa): Advanced stuff covered here, but David is a fantastic lecturer and I loved the comprehensive content. 
	* [Simple Reinforcement Learning with Tensorflow](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0): Arthur Juliani has a blog post series that covers RL concepts with lots of practical examples. 

Awesome, so now you should have a decent understanding of where ML and DL are in today's day and age. The world is your playground now. Read research papers, try Kaggle contests, watch ML tech talks, build cool projects, talk to others interested in ML, never stop learning, and most importantly, have fun! :) This is a great time to get into ML and in the rush to gain knowledge as quickly as possible, it often good to just slow down and think about the types of amazing applications and positive change we can create in this world with this technology.  

## Best Courses

* [Stanford CS 231N](https://www.youtube.com/watch?v=g-PvXUjD6qg&list=PLlJy-eBtNFt6EuMxFYRiNRS07MCWN5UIA) - CNNs
* [Stanford CS 224D](https://www.youtube.com/watch?v=sU_Yu_USrNc&list=PLTuSSFCVeNVCXL0Tak5rJ83O-Bg_ajO5B) - Deep Learning for NLP
* [Hugo Larochelle's Neural Networks Course](https://www.youtube.com/watch?v=SGZ6BttHMPw&list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
* [David Silver's Reinforcement Learning Course](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PL5X3mDkKaJrL42i_jhE4N-p6E2Ol62Ofa)
* [Andrew Ng Machine Learning Course](https://www.coursera.org/learn/machine-learning)
* [Stanford CS 229](https://www.youtube.com/playlist?list=PLA89DCFA6ADACE599) - Pretty much the same as the Coursera course
* [UC Berkeley Kaggle Decal](https://kaggledecal.github.io/)
* [Short MIT Intro to DL Course](https://www.youtube.com/playlist?list=PLkkuNyzb8LmxFutYuPA7B4oiMn6cjD6Rs)
* [Udacity Deep Learning](https://www.udacity.com/course/deep-learning--ud730)
* [Deep Learning School Montreal 2016](http://videolectures.net/deeplearning2016_montreal/) and [2017](http://videolectures.net/deeplearning2017_montreal/)
* [Intro to Neural Nets and ML (Univ of Toronto)](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2017/)
* [Deep RL Bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures)
* [CMU Neural Networks for NLP](http://www.phontron.com/class/nn4nlp2017/schedule.html)
* [Bay Area Deep Learning School Day 1 2016](https://www.youtube.com/watch?v=eyovmAtoUx0) and [Day 2](https://www.youtube.com/watch?v=9dXiAecyJrY)
* [Introduction to Deep Learning MIT Course](https://www.youtube.com/playlist?list=PLkkuNyzb8LmxFutYuPA7B4oiMn6cjD6Rs)
* [Caltech CS 156 - Machine Learning](https://www.youtube.com/playlist?list=PLD63A284B7615313A)
* [Berkeley EE 227C - Convex Optimization](https://ee227c.github.io/)

## Most Important Deep Learning Papers

Not a comprehensive list by any sense. I just thought these papers were incredibly influential in getting deep learning to where it is today. 

* [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
* [GoogLeNet](https://arxiv.org/pdf/1409.4842v1.pdf)
* [VGGNet](https://arxiv.org/pdf/1409.1556v6.pdf)
* [ZFNet](https://arxiv.org/pdf/1311.2901v3.pdf)
* [ResNet](https://arxiv.org/pdf/1512.03385.pdf)
* [R-CNN](https://arxiv.org/pdf/1311.2524v5.pdf)
* [Fast R-CNN](https://arxiv.org/pdf/1504.08083.pdf)
* [Adversarial Images](https://arxiv.org/pdf/1412.1897.pdf)
* [Generative Adversarial Networks](https://arxiv.org/pdf/1406.2661v1.pdf)
* [Spatial Transformer Networks](https://arxiv.org/pdf/1506.02025.pdf)
* [DCGAN](https://arxiv.org/pdf/1511.06434v2.pdf)
* [Synthetic Gradients](https://arxiv.org/pdf/1608.05343v1.pdf)
* [Memory Networks](https://arxiv.org/pdf/1410.3916v11.pdf)
* [Mixture of Experts](https://arxiv.org/pdf/1701.06538.pdf)
* [Neural Turing Machines](https://arxiv.org/pdf/1410.5401.pdf)
* [Alpha Go](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf)
* [Atari DQN](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
* [Word2Vec](https://arxiv.org/pdf/1310.4546.pdf)
* [GloVe](https://nlp.stanford.edu/pubs/glove.pdf)
* [A3C](https://arxiv.org/pdf/1602.01783v2.pdf)
* [Gradient Descent by Gradient Descent](https://arxiv.org/pdf/1606.04474v1.pdf)
* [Rethinking Generalization](https://arxiv.org/pdf/1611.03530v1.pdf)
* [Densely Connected CNNs](https://arxiv.org/pdf/1608.06993v1.pdf)
* [EBGAN](https://arxiv.org/pdf/1609.03126v1.pdf)
* [Wasserstein GAN](https://arxiv.org/pdf/1701.07875.pdf)
* [Style Transfer](https://arxiv.org/pdf/1603.08155v1.pdf)
* [Pixel RNN](https://arxiv.org/pdf/1601.06759v2.pdf)
* [Dynamic Coattention Networks](https://arxiv.org/pdf/1611.01604v2.pdf)
* [Convolutional Seq2Seq Learning](https://arxiv.org/pdf/1705.03122.pdf)
* [Seq2Seq](https://arxiv.org/pdf/1409.3215.pdf)
* [Dropout](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)
* [Batch Norm](https://arxiv.org/pdf/1502.03167.pdf)
* [Large Batch Training](https://arxiv.org/pdf/1609.04836.pdf)
* [Transfer Learning](http://papers.nips.cc/paper/5347-how-transferable-are-features-in-deep-neural-networks.pdf)
* [Adam](https://arxiv.org/pdf/1412.6980.pdf)
* [Speech Recognition](https://arxiv.org/pdf/1303.5778.pdf)
* [Relational Networks](https://arxiv.org/pdf/1706.01427.pdf)
* [Influence Functions](https://arxiv.org/pdf/1703.04730.pdf)
* [ReLu](https://arxiv.org/pdf/1611.01491.pdf)
* [Xavier Initialization](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
* [Saddle Points and Non-convexity of Neural Networks](https://arxiv.org/pdf/1406.2572.pdf)
* [Overcoming Catastrophic Forgetting in NNs](https://arxiv.org/pdf/1612.00796.pdf)
* [Quasi-Recurrent Neural Networks](https://arxiv.org/pdf/1611.01576.pdf)
* [Escaping Saddle Points Efficiently](https://arxiv.org/pdf/1703.00887.pdf)
* [Progressive Growing of GANs](http://research.nvidia.com/sites/default/files/pubs/2017-10_Progressive-Growing-of//karras2017gan-paper.pdf)
* [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
* [Dynamic Routing Between Capsules](https://arxiv.org/pdf/1710.09829.pdf)
* [Unsupervised Machine Translation with Monolingual Corpora](https://arxiv.org/pdf/1711.00043.pdf)
* [Population Based Training of NN's](https://arxiv.org/pdf/1711.09846.pdf)
* [Learned Index Structures](https://arxiv.org/pdf/1712.01208v1.pdf)
* [Visualizing Loss Landscapes](https://arxiv.org/pdf/1712.09913v1.pdf)
* [DenseNet](https://arxiv.org/pdf/1608.06993.pdf)
* [SqueezeNet](https://arxiv.org/pdf/1602.07360.pdf)

## Cool Use Cases of ML

* [Dropbox OCR](https://blogs.dropbox.com/tech/2017/04/creating-a-modern-ocr-pipeline-using-computer-vision-and-deep-learning/)
* [Blend Application Submission RNNs](https://blend.com/predicting-submission/)
* [Uber Customer Care NLP](https://eng.uber.com/cota/)

## ML Tech Talks

* [Democratizing Machine Learning - Neil Lawrence](https://www.youtube.com/watch?v=2Shx0cW1bMI)
* [Deep Learning for NLP - Richard Socher](https://www.youtube.com/watch?v=tdLmf8t4oqM)
* [2015 ICML Deep Learning Panel Discussion](https://www.youtube.com/watch?v=EiStan9i8vA)
* [Practical Application of AI in Enterprise](https://www.youtube.com/watch?v=6LU5Rd587Vk)
* [Using AI to Personlize Healthcare](https://www.youtube.com/watch?v=KB45rBClgmA)
* [Autonomous Driving CVPR 2016 Keynote - Amnon Shashua](https://www.youtube.com/watch?v=n8T7A3wqH3Q)
* [From Text to Knowledge via ML - Xavier Amatriain](https://www.youtube.com/watch?v=q_heZNJ4blA)
* [Deep Visual-Semantic Alignments for Generating Image Descriptions](http://techtalks.tv/talks/deep-visual-semantic-alignments-for-generating-image-descriptions/61593/)
* [Deep Learning Theoretical Motivations - Yoshua Bengio](http://videolectures.net/deeplearning2015_bengio_theoretical_motivations/?q=bengio)
* [Learning to Learn and Compositionality with RNNs - Nando de Freitas](https://www.youtube.com/watch?v=x1kf4Zojtb0)
* [Deep Advances in Generative Modeling - Alec Radford](https://www.youtube.com/watch?v=KeJINHjyzOU)
* [Adversarial Networks - Soumith Chintala](https://www.youtube.com/watch?v=QPkb5VcgXAM)
* [The Next Frontier in AI: Unsupervised Learning - Yann LeCun](https://www.youtube.com/watch?v=IbjF5VjniVE)
* [ML Foundations and Methods for Precision Medicine and Healthcare - Suchi Saria](https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/ML-Foundations-and-Methods-for-Precision-Medicine-and-Healthcare)
* [Reprogramming the Human Genome: Why AI is Needed - Brendan Frey](https://www.youtube.com/watch?v=dcU93uo1qu0)
* [Virtual Assistant: Smart Reply - Anjuli Kannan](https://www.youtube.com/watch?v=ipzBmb4WLAk)
* [Bridging the Gap Between Artificial Intelligence and Neuroscience - Alex Lavin](https://www.youtube.com/watch?v=7kiTiiMSCrI)
* [Bringing Deep Learning to The Front Lines of Healthcare - Will Jack](https://www.youtube.com/watch?v=bG0ZbRCYCo8)
* [Deep Robotic Learning - Sergey Levine](https://www.youtube.com/watch?v=eKaYnXQUb2g)
* [How AI Startups Must Compete with Google - Fei-Fei Li](https://www.youtube.com/watch?v=Mu3scWZvZKo)
* [Past, Present and Future of AI Panel](https://www.youtube.com/watch?v=0ueamFGdOpA&list=PLOU2XLYxmsIKC8eODk_RNCWv3fBcLvMMy&index=118)
* [Differentiable Neural Computer - Alex Graves](https://www.youtube.com/watch?v=steioHoiEms)
* [Technology, Policy and Vehicle Safety in the Age of AI - Chris Gerdes](https://www.youtube.com/watch?time_continue=990&v=LDprUza7yT4)
* [Attention and Memory in Deep Learning Networks - Stephen Merity](https://www.youtube.com/watch?v=uuPZFWJ-4bE)
* [Deep Learning in the Brain - Blake Richards](http://videolectures.net/deeplearning2017_richards_neuroscience/)
* [Building Machines that See, Learn, and Think Like People - Josh Tenenbaum](https://www.youtube.com/watch?v=7ROelYvo8f0)
* [Self-Driving Cars - Sacha Arnoud](https://www.youtube.com/watch?v=LSX3qdy0dFg)
* [Deep Learning and Innate Priors - Yann LeCun and Christopher Manning](https://www.youtube.com/watch?v=fKk9KhGRBdI)
* [nuTonomy's Vision for Autonomous Driving - Emilio Frazzoli](https://www.youtube.com/watch?v=dWSbItd0HEA)
* [Meta-Learning and Self-Play - Ilya Sutskever](https://www.youtube.com/watch?v=9EN_HoEk3KY)

## Best Blogs

* [Andrej Karpathy](http://karpathy.github.io/)
* [Google Research](https://research.googleblog.com/)
* [Neil Lawrence](http://inverseprobability.com/blog)
* [Qure.ai](http://blog.qure.ai/)
* [Brandon Amos](http://bamos.github.io/blog/)
* [Denny Britz](http://www.wildml.com/)
* [Moritz Hardt](http://blog.mrtz.org/)
* [Deepmind](https://deepmind.com/blog/)
* [Machine Learning Mastery](http://machinelearningmastery.com/blog/)
* [Smerity](http://smerity.com/articles/articles.html)
* [The Neural Perspective](https://theneuralperspective.com/)
* [Pete Warden](https://petewarden.com/page/2/)
* [Kevin Zakka](https://kevinzakka.github.io/)
* [Thomas Dinsmore](https://thomaswdinsmore.com/)
* [Rohan Varma](http://rohanvarma.me/)
* [Anish Athalye](https://www.anishathalye.com/)
* [Arthur Juliani](https://medium.com/@awjuliani)
* [CleverHans](http://www.cleverhans.io/)
* [Off the Convex Path](http://www.offconvex.org/about/)
* [Sebastian Ruder](http://ruder.io/#open)
* [Berkeley AI Research](http://bair.berkeley.edu/blog/)
* [Facebook AI Research](https://research.fb.com/blog/)
* [Salesforce Research](https://www.salesforce.com/products/einstein/ai-research/)
* [Apple Machine Learning Journal](https://machinelearning.apple.com/)
* [OpenAI](https://blog.openai.com/)
* [Lab41](https://gab41.lab41.org/tagged/machine-learning)
* [Distill](https://distill.pub/)
* [My blog :)](https://adeshpande3.github.io/)

## Data and Features

* Normalizing inputs isn’t guaranteed to help. 
    * A lot of people say you should always normalize, but TBH in practice, it doesn’t help for me all the time. This advice is very very dataset dependent, and it’s up to you to figure out if normalizing will be helpful for your particular task. 
    	* It's also model dependent. Normalization will likely help for linear models like linear/logistic reg, but won't be as helpful for deep learning models and neural networks. 
    * In order to determine whether it will help or not, visualize your data to see what type of features you're dealing with. Do you have some features that have values around 1000 or 2000? Do you have some that are only 0.1 or 0.2? If so, normalizing those inputs is likely a good idea. 
* The process in which you convert categorical features into numerical ones is very important. 
    * You have to decide whether you want to use dummies variables, use bins, etc.
* Many different methods of normalization
    * Min-max normalization where you do (x - a)/(b-a) where x is the data point, a is the min value and b is the max value. 
    * Max normalization where you do x/b where b is the max value. 
    * L1 normalization where you do x/c where c is the summation of all the values. 
    * L2 normalization where you do x/c where c is the square root of the summation of all the squared values.
    * Z score normalization where you do (x - d)/e where d is the mean and e is the standard deviation.
* Something interesting you can do if you either have little data or missing data is that you can use a neural network to predict those values based on the data that you currently have. 
	* I believe the matrix imputation is the correct term. 
* Data augmentation comes in many different forms. Noise injection is the most general form. While working with images, the number of options increases a whole lot.
    * Translations
    * Rotations
    * Brightness
    * Shears
* Whenever thinking about whether or not to even apply machine learning to your problem, always consider the type of data that you have to solve the problem. Let's consider a model where you want to predict whether someone will develop cancer in the next few months based on their health data. The data is your input into the model and the output is a binary result. Seems like a good place where we can apply machine learning right, given that we have a number of patients' data and cancer results? However, we should also be thinking about whether the health data that we have about the individual is *enough* to be able to make some conclusion about the patient. Does the health data accurately encompass the different amounts of variation there are from person to person? Are we accurately representing each input example with the most comprehensive amount of information? Because if the data doesn't have enough information/features, your model will risk just overfitting to noise or the training set, and won't really learn anything useful at all. 
* Never underestimate the power of simply increasing the size of your training dataset when you're attempting to improve your model accuracy. If you have $100 for your machine learning project, spend like 90 of those dollars on good data collection and preprocessing.

## Models

* For regression problems where the goal is to output some real valued number, linear regression should always be your first choice. It gets great accuracy on a lot of datasets. Obviously, this is very dependent on the type of data you have, but you should always first try linreg, and then move on to other more complex models.
* XGBoost has been one of the best models I've seen for a lot of regression or classification tasks. 
	* Averaging multiple runs of XGBoost with different seeds helps to reduce model variance. 
    * Model hyperparameter tuning is important for XGBoost
* Whenever your features are highly correlated with each other, running OLS will have a large variance, and therefore you should use ridge regression instead.
* KNN does poorly with high dimensional data because the distance metrics would get all messed up with the scales.
* One of the themes I hear over and over again is that the effective capacity of your model should match the complexity of your task. 
    * For simple house price prediction, a linear regression model might do. 
    * For object segmentation, a large and specialized CNN will be a necessity. 
* Use RNNs for time series data and natural language data since NLP data is contains dependencies between different steps in the input.

## Hyperparameters

* Optimizing hyperparameters can be a bit of a pain and can be really computationally expensive, but sometimes they can make anywhere from a 1 - 5 percent improvement. In some use cases, that can be the difference between a good product and a great product. 
* Not saying that everyone should invest in a Titan X GPU to do large grid searches for optimizing hyperparameters, but it's very important to develop a structured way for how your going to go about it. 
	* The easiest and simplest way is to just write a nested for loop that goes through different values, and keeps track of the models that perform the best. 
	* You can also use specialized libraries and functions that will make the code look a little nicer. 
		* Scikit-Learn has a good [one](http://scikit-learn.org/stable/modules/grid_search.html)
* SVM’s
    * [What are C and Gamma?](https://www.quora.com/What-are-C-and-gamma-with-regards-to-a-support-vector-machine/answer/Rohan-Varma-8?srid=JlQJ) 
* The Deep Learning book says that the learning rate is the most important hyperparameter in most deep networks, and I do have to agree. A low LR makes training unbearably slow (and could get stuck in some minima), while a high LR can make training extremely unstable. 

## Tensorflow/Deep Nets

* Always make sure your model is able to overfit to a couple pieces of the training data. Once you know that the network is learning correctly and is able to overfit, then you can work on building a more general model.
* Evaluation of your Tensorflow graph only happens at runtime. Your job is to first define the variables and the placeholders and the operations (which together make up the graph) and then you can execute this graph in a session (execution environment), where you feed in values into the placeholders. 
* Regularization is huuuuugely important with deep nets. 
    * Weight Decay
    * Dropout
        * Seriously. Use it. It’s sooo simple, but it’s so good at helping to prevent overfitting and thus gives your test accuracy a nice boost most of the time. 
    * Data Augmentation
    * Early Stopping
        * Really effective. Basically, just stop training when the validation error stops decreasing. Because if you training for any longer, you’re just going to overfit to the training dataset more, and that’s not going to help you with that test set accuracy. And honestly, this method of regularization is just really easy to implement.  
    * Batch normalization (Main purpose is not necessarily to regularize, but there's a slight regularization effect)
    	* In my experience, batch norm is very helpful with deep nets. The intuition is that it helps reduce the internal covariate shift inside the network. In simpler terms, let's think about the activations in layer 3 of the network. These activations are a function of the weights of the previous two layers and the inputs. The job of the 3rd layer weights is to take those activations and apply another transformation such that the resulting output is close to the predicted value. Now, because the activations are a function of the weight matrices of the previous two layers (which are changing a lot b/c gradient descent), the distribution of those activations can vary a lot as the network is training. Intuitively, this makes it very difficult for the weights of the 3rd layer to figure out how to properly set itself. Batch norm helps alleviate that problem that making sure to keep the mean and variance of the activations the same (the exact values for mean and variance aren't necessarily 0 and 1, but are rather a function of 2 learnable parameters beta and gamma). This results in the activation distributions changing a lot less, meaning that the weights of the 3rd layer are easier to get adjusted to the optimal values, and thus results in faster training times. 
		* Great [blog post](http://rohanvarma.me/Batch-Norm/) by Rohan Varma
    * Label smoothing
    * Model averaging
    * Regularizers are thought to helpful the generalization ability of networks, but are not the *main* reason that deep nets generalize so well. 
The idea is that you don’t use all of the above, but if you are having overfitting issues, you should use some of these to combat those issues. 
* That being said, modal architecture and the structure of your layers is sometimes thought to be more important. 
* Always always always do controlled experiments. By this, I mean only change one variable at a time when you’re trying to tune your model. There is an incredible number of hyperparameters and design decisions that you can change. Here are a few examples.
    * Network architecture
        * Number of convolutional layers in a CNN
        * Number of LSTM units in an RNN
    * Choice of optimizer
    * Learning rate
    * Regularization
    * Data augmentation
    * Data preprocessing
    * Batch size
If you try to change too many of the above variables at once, you’ll lose track of which changes really had the most impact. Once you make changes to the variables, always keep track of what impact that had on the overall accuracy or whatever other metric you’re using. 

## Deep Learning Frameworks

* [Keras](https://keras.io/) - My friend and I have a joke where we say that you’ll have a greater number of lines in your code just doing Keras imports, compared to the actual code because the functions are so incredibly high level. Like seriously. You could load a pretrained network and finetune it on your own task in like 6 lines. It’s incredible. This is definitely the framework I use for hackathons and when I’m in a time crunch, but I think if you really really want to learn ML and DL, relying on Keras’s nice API might not be the best call. 
* [Tensorflow](https://www.tensorflow.org/) - This is my go-to deep library nowdays. Honestly I think it has the steepest learning curve because it takes quite a while to get comfortable with the ideas of Tensorflow variables, placeholders, and building/executing graphs. One of the big plus sides to Tensorflow is the number of Github and Stackoverflow help you can get. You can find the answer to almost any error you get in Tensorflow because someone has likely run into it before. I think that's hugely helpful. 
* [Torch](http://torch.ch/) - 2015 was definitely the year of Torch, but unless you really want to learn Lua, PyTorch is probably the way to go now. However, there’s a lot of good documentation and tutorials associated with Torch, so that’s a good upside. 
* [PyTorch](http://pytorch.org/) - My other friend and I have this joke where we say that if you’re running into a bug in PyTorch, you could probably read the entirety of PyTorch’s documentation in less than 2 hours and you still wouldn’t find your answer LOL. But honestly, so many AI researchers have been raving about it, so it’s definitely worth giving it a shot even though it’s still pretty young. I think Tensorflow and PyTorch will be the 2 frameworks that will start to take over the DL framework space.
* [Caffe](http://caffe.berkeleyvision.org/) and [Caffe2](https://caffe2.ai/) - Never played around with Caffe, but this was one of the first deep learning libraries out there. Caffe2 is notable because it's the production framework that Facebook uses to serve its models. [According to Soumith Chintala](https://www.oreilly.com/ideas/why-ai-and-machine-learning-researchers-are-beginning-to-embrace-pytorch), researchers at Facebook will try out new models and research ideas using PyTorch and will deploy using Caffe2. 

## CNNs

* Use CNNs for *any* image related task. It's really hard for me to think of any image processing task that hasn't been absoluted revolutionized by CNNs. 
	* That being said, you might not want to use CNNs if you have latency, power, computation, or memory constraints. In a lot of different areas (IoT for example), some of the downsides of using a CNN flare up. 
* Transfer learning is harder than it looks. Found out from firsthand experience. During a hackathon, my friend and I wanted to determine whether someone has bad posture or not (from a picture) and so we spent quite a bit of time creating a 500 image dataset, but even with using a pretrained model, chopping off the last layer, and retraining it, and while the network was able to learn and get a decent accuracy on the training set, the validation and testing accuracies weren’t up to par signaling that overfitting might be a problem (due to our small dataset). Moral of the story is don’t think that transfer learning will come and save your image processing task. Take a good amount of time to create a solid dataset, and understand what type of model you’ll need, and what kind of modifications you’ll need to make to the pretrained network. 
	* Transfer learning is also interesting because there are two different ways that you can go with it. You can use transfer learning for finetuning. This is where you take a pretrained CNN (trained on Imagenet), chop off the last fully connected layer, add an FC layer specific to your task, and then retrain on your dataset. However, there's also another interesting approach called transfer learning for feature extraction. This is where you take a pretrained CNN, pass your data through the CNN, get the output activations from the last convolutional layer, and then use that feature representation as your data for training a more simple model like an SVM or linear regression.  

## NLP

* Use pretrained word embeddings whenever you can. From my experience, it’s just a lot less hassle, and quite frankly I’m not sure if you even get a performance improvement if you try to train them jointly with whatever other main task you want to solve. It’s task-dependent I guess. For something simple like sentiment analysis though, pretrained word vectors worked perfectly. 
	* [Glove word embeddings](https://nlp.stanford.edu/projects/glove/) work great for me. 

## Deep Reinforcement Learning

Some cool articles and blog posts on RL. 
* [Deep Reinforcement Learning Doesn't Work Yet](https://www.alexirpan.com/2018/02/14/rl-hard.html)
* [Deep RL Arxiv Review Paper](https://arxiv.org/pdf/1701.07274v2.pdf)
* [Pong From Pixels](http://karpathy.github.io/2016/05/31/rl/)
* [Lessons Learned Reproducing a Deep RL Paper](http://amid.fish/reproducing-deep-rl)

## ML Project Advice

* Create your machine learning pipeline first and then start to worry about how you can tune and make your model better later. 
    * For example, if you’re creating an image classification model, make sure you’re able to load data into your program, create test/train matrices, create a very simple model (using one of the DL libraries), create your training loop, and make sure the network is learning something and that you can get to some baseline accuracy. Only once you’ve done these steps can you start to worry about things like regularization, data augmentation, etc. 
    * Too many times you can get over complicated with the model and hyperparameters when your issues may lie just in the way you’re loading in data or creating your training batches. Be sure to get those simple parts of the machine learning pipeline down. 
    * Another benefit to making sure you have a minimal but working end to end pipeline is that you’re able to track performance metrics as you start to change your model and tune your hyperparameters. 
* If you’re trying to create some sort of end product, 80% of your ML project will honestly be just doing front-end and back-end work. And even within that 20% of ML work, a lot of it will probably be dataset creation or preprocessing. 
* Always divide your data into train and validation (and test if you want) sets. Checking performance on your validation set at certain points during training will help you determine whether the network is learning and when overfitting starts to happen.  
* Always shuffle your data when you're creating training batches. 

## Math Resources

* [Matrix Calculus](http://parrt.cs.usfca.edu/doc/matrix-calculus/index.html)
* [Mathematics for Machine Learning](http://gwthomas.github.io/docs/math4ml.pdf)

## Bias in Machine Learning

* [CS 294: Fairness in Machine Learning](https://fairmlclass.github.io/)
* [Bias and Fairness in Machine Learning Blog Post](https://abhishek-tiwari.com/bias-and-fairness-in-machine-learning/)
* [Fairness in Precision Medicine](https://datasociety.net/wp-content/uploads/2018/02/Data.Society.Fairness.In_.Precision.Medicine.Feb2018.FINAL-2.26.18.pdf)

## Kaggle

* Bit of a love/hate relationship with Kaggle. I think it's great for beginners in machine learning who are looking to get more practical experience. I can't tell you how much it helped me to go through the process of loading in data with Pandas, creating a model with Tensorflow/Scikit-Learn, training the model, and fine tuning to get good performance. Seeing how your model stacks up against the competition afterwards is a cool feeling as well. 
* The part of Kaggle that I don't really enjoy is how much feature engineering is required to really get into the top 10-15% of the leaderboard. You have to be really committed to data visualization and hyperparameter tuning. Some may argue that this is a crucial part of being a data scientist/machine learning engineer, and I do agree that it's important in real world problem spaces. I think the main point here is that if you want to get good at Kaggle competitions, there's no easy road, and you'll have to do a lot of practice (which is not bad!).
* If you're looking to get better at Kaggle competitions, I'd recommend reading their [blog](http://blog.kaggle.com/). They interview a lot of the competition winners, which will give you a good look at the type of ML models and pipelines needed to get good performance.   

## Debugging ML Models

* What should you do when your ML model’s accuracy just isn’t good enough?
* If your train accuracy is great but your validation accuracy is poor, this likely means that your model has overfit to the training data.
    * Reduce model complexity
    * Gather more data
        * Data augmentation helps (at least for image data)
        * Honestly, just investing more time/money in this component of your ML model is almost always well worth it.
    * Regularization
* If your training accuracy just isn't good enough, you could be suffering from underfitting.
    * Increase model complexity
        * More layers or more units in each layer
	* Don't try to add more data. Your model clearly isn't able to classify the examples in the current dataset, so adding more data won't help.
* Any other problem you might be having.
    * Check the quality of your training data and make sure you’re loading everything in properly
    * Adjust hyperparameters
        * I know it’s tedious and it takes a boatload of time, but it can *sometimes* be worth. 
        * Grid search might be the best approach, but keep in mind it can get extremely computationally expensive.
        * I heard random search is also surprisingly effective, but haven’t tried it myself. 
    * Restart with a very simple model and only a couple of training points, and make sure your model is able to learn that data. Once it gets 100% accuracy, start increasing the complexity of the model, as well as loading in more and more of your whole dataset. 

## Best Python Libraries for ML

* [Numpy](http://www.numpy.org/)
* [Scikit-Learn](http://scikit-learn.org/stable/)
* [Matplotlib](https://matplotlib.org/)
* [Pandas](http://pandas.pydata.org/)

## Other Interesting Links

* [Neural Network Playground](http://playground.tensorflow.org)
* [ConvNetJS Demo](https://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html)
* [Stanford Sentiment Analysis Demo](http://nlp.stanford.edu:8080/sentiment/rntnDemo.html)
* [Deep Learning's Relation to Physics](https://www.technologyreview.com/s/602344/the-extraordinary-link-between-deep-neural-networks-and-the-nature-of-the-universe/)
* [Interpretability of AI](http://nautil.us/issue/40/learning/is-artificial-intelligence-permanently-inscrutable)
* [Picking up Deep Learning as an Engineer](https://www.quora.com/What-are-the-best-ways-to-pick-up-Deep-Learning-skills-as-an-engineer/answer/Greg-Brockman?srid=cgo&share=d1ac0da2)
* [Feature Visualization](https://distill.pub/2017/feature-visualization/)
* [Expressivity, Trainability, and Generalization in Machine Learning](http://blog.evjang.com/2017/11/exp-train-gen.html?spref=tw)
* [Optimization over Explanation](https://medium.com/berkman-klein-center/optimization-over-explanation-41ecb135763d)
* [Learning to Trade with RL](http://www.wildml.com/2018/02/introduction-to-learning-to-trade-with-reinforcement-learning/)
* [AI and Digital Forgery](https://giorgiop.github.io/posts/2018/03/17/AI-and-digital-forgery/)
* [Judea Pearl Interview](https://www.quantamagazine.org/to-build-truly-intelligent-machines-teach-them-cause-and-effect-20180515/)

## UCLA ACM AI Resources

As one of the officers for UCLA ACM AI, the AI/ML group on campus, we have hosted several workshops on introducing students to machine learning. Here are the resources from our Intro to ML with Tensorflow workshop series. 

* [Workshop #1: Linear Regression Slides](http://bit.ly/2hj5epX)
* [Workshop #1: Linear Regression Code](https://github.com/uclaacmai/tf-workshop-series-fall17/blob/master/week2-linear-regression/Intro%20to%20TF%20and%20Linear%20Regression.ipynb)
* [Workshop #2: Logistic Regression Slides](http://bit.ly/2ybSdZW)
* [Workshop #2: Logistic Regression Code](https://github.com/uclaacmai/tf-workshop-series-fall17/blob/master/week3-classification/Classification%20and%20Logistic%20Regression.ipynb)
* [Workshop #3: Pandas & Kaggle Hack Session Slides](http://bit.ly/2zN49gV)
* [Workshop #3: Pandas & Kaggle Hack Session Code](https://github.com/uclaacmai/tf-workshop-series-fall17/blob/master/week4-pandas-and-kaggle/Kaggle%20Hack%20Session%20Starter.ipynb)
* [Workshop #4: Neural Networks with Tensorflow Slides](http://bit.ly/2yqN2W9)
* [Workshop #4: Neural Networks with Tensorflow Code](https://github.com/uclaacmai/tf-workshop-series-fall17/blob/master/week5-neural-networks/Neural%20Network%20Full%20Tutorial.ipynb)
* [Workshop #5: Multi-Layer Neural Networks Slides](http://bit.ly/2zwwgUM)
* [Workshop #5: Multi-Layer Neural Networks Code](https://github.com/uclaacmai/tf-workshop-series-fall17/blob/master/week6-multi-layer-neural-networks/Multi-Layer%20NN%20MNIST.ipynb)
* [Workshop #6: Convolutional Neural Networks Slides](https://docs.google.com/presentation/d/12uEAa9Tpm660ggIVKtoPpyhVC_N41uVO5itMNsXA1Y8/edit?usp=sharing)
* [Workshop #6: Convolutional Neural Networks Code](https://github.com/uclaacmai/tf-workshop-series-fall17/blob/master/week7-convolutional-neural-networks/CNN%20Full%20Tutorial.ipynb)
* [Workshop #7: Recurrent Neural Networks Slides](https://docs.google.com/presentation/d/1MmU6mNhX0fuOE_z02bTtiXlgslRYSljuaU2OMatM7bg/edit?usp=sharing)
* [Workshop #7: Recurrent Neural Networks Code](https://github.com/uclaacmai/tf-workshop-series-fall17/blob/master/week8-recurrent-neural-networks/Recurrent%20Neural%20Networks.ipynb)

## Random Thoughts

* It's interesting and worrying that the process in which neural nets and deep learning work is still such a black box. If an algorithm comes to the conclusion that a number is let's say classified as a 7, we don't really know how the algorithm came to that result because it's not hardcoded anywhere. It's hidden behind millions of gradients and derivatives. So if we wanted to use deep learning in a more important field like medicine, the doctor who is using this technology should be able to know why the algorithm is giving the result it's giving. Doctors also generally don’t like the idea of probabilistic results.
    * "People still don’t trust black-boxes. They need to see the audit trails on how decisions were made, something most AI technologies can’t furnish."
    * Medical Diagnosis, Insurance claim rejections, and loan/mortgage applications. 
        * You can’t just have a ML system that basically outputs a binary response for whether or not someone has a certain disease without being able to point to some feature in the input data that explains the corresponding output. 
        * Yoshua Bengio, though, argues that all you need is statistical reassurance about the general ability of the trained model.  
    * Jeff Dean mentioned that this interpretability problem is an issue in some domains and not an issue with others. I think that’s the way you have to look at this topic, it’s should be a case-by-case basis and we shouldn’t really make broad statements saying lack of interpretability is terrible or lack of interpretability is something we can deal with. 
* If you have problems where you can generate a lot of synthetic data, for example with reinforcement learning with arcade games. You can simulate so many rounds of games and get so much data off of that for pretty much nothing. 
* Important to differentiate between which problems just need more data and compute to work better and which problems really need core algorithmic breakthroughs to really work. 
* Breakthroughs over the past 5 or so years have mostly happened because of improvements in data/compute/infrastructure, not necessarily improvements in the algorithms themselves. At some point, we're going to have to make those better. Compute and data isn't going to be enough at some point.
* How is there is a trust between the human and the machine. In the past, machines would work exactly the way it is supposed to (as long as servers don't go down), This is kind of a new frontier where you don’t really know that given a certain input, what the output will look like. 
* The main reasons that deep learning methods have really worked over the last few years are:
    * The public availability of large datasets. Aka Thank you Fei-Fei Li!
    * Compute power. Aka Thanks Nvidia
    * Interesting models that take advantage of certain characteristics in data.
        * CNNs using convolutional filters to analyze images.
        * RNNs using recurrent networks to exploit sequence dependence in natural language data.

## Research Ideas

* Creating systems that don’t need as much training data to be effective.
    * One shot and zero shot learning.
* Networks that have specialized memory units that hold specific past data points and their labels.
    * Neural Turing Machine and Differentiable Neural Computer have external memories.
* Determining the actually useful things that GANs can do.
    * Generating synthetic data
    * Usage as a feature extractor to understand more about your data’s distribution in an unsupervised learning setting (specifically one where you have a lot of unlabeled data), and then using those features for a different supervised learning task.
* Creating models with less labeled training data using one shot learning, learning from unstructured data, weak labels,
* Making reinforcement learning algorithms better in environments with sparse reward signals.
* How to make computations faster on hardware? How to make memory storage and access faster? How does everything change when you try distributing the workload to lots of different servers?
* Multitask learning in order to have single networks that able to solve a bunch of different tasks given input, rather than having one (CNN) for image inputs and one (RNN) for language inputs. 
    * The main thing here is that we’d like to be able to compute some shared representation of the input so that we’re able to do some sort of analysis of it, regardless of whether the input is an image or text or speech, etc.
* Adding more variety to your loss function. A traditional loss function just represents the difference between a network’s prediction and the true label, and our optimization procedure seeks to minimize that. We can also try to get creative with our loss functions and add soft constraints on the values of the weights (weight decay) or the values of the activations, or honestly whatever desirable property we want in our model.

## Other

* Jupyter Notebook is the **best**. Seriously. It’s so nice for ML because visualizing data and understanding dimensionality of your variables is so vital, and having an environment where you can do both of those things easily is very helpful. 
* The number of clusters that you choose in K nearest neighbors is a hidden latent variable because it is something that isn’t observed in the actual dataset. 
* We use dimensionality reduction because data may appear high dimensional but there may only be a small number of degrees of variability.  
* Don’t bother with unsupervised learning unless you have a really simple task and you want to use K-Means, or if you’re using PCA to reduce the dimensionality of data, or if you’re playing around with autoencoders. PCA can be ridiculously useful sometimes.
	* That being said, unsupervised methods can be helpful in scenarios where it is costly/impossible to obtain labeled data. In these cases, it might be beneficial to use some sort of clustering or PCA to do analysis on your data instead of relying on a supervised learning ML approach which may not be feasible. 
	* My main point is that trying to frame your problem as a supervised learning one, and investing some time in trying to obtain labels is likely to be worth it, because I just don't think unsupervised learning is at a stage where you . This decision of supervised vs unsupervised is very task dependent though, so I'm hesitant of making a catch-all statement.
* Model compression is something that becomes extremely important when you’re trying to deploy pretrained models onto mobile devices. Training a model offline with GPUs and then making sure that the network size is small enough so that end users can just run inference on those models at test time. Most of the time, the state of the art CNN models (like Inception or ResNet) are so huge that all the parameters can’t possibly fit on one device. That’s where model compression comes in. 
* Haven’t played around a whole lot with auto-encoders, but they seem really interesting. The basic idea is that you want a network that maps your input into another representation h, and then from h back to your original input. This is interesting because throughout the process of training, the weight values in between your x and h representations could give us a lot of good information about the type of data distribution we have in our training set. A really cool unsupervised learning method. 
