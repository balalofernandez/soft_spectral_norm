# Soft spectral normalization
A parametrized way of imposing soft_spectral normalization in pytorch.

Spectral normalization was a method originally proposed to improve training in Generative Adversarial Networks (GANs) to address the mode collapse issue. Mode collapse is simply a phenomenon that occurs when a generative model's outputs become overly homogeneous, failing to capture the full range of the true data distribution. In the context of images, the generator will produce a narrow range of highly similar results, which is undesired.

Spectral normalization simply makes training more stable by controlling the Lipschitz constant of the network. This Lipschitz property guarantees a smooth and sensitive function, which is often desirable for creating a measure of distance between inputs and hidden states of the network denoted by $f$.
$$
L_{1} * ||x_{1} - x_{2}||_{X} \leq || f(x_{1}) - f(x_{2})||_{F} \leq L_{2} * ||x_{1} - x_{2}||_{X}
$$
Another way of enforcing bi-Lipschitzness is a two-sided gradient penalty where you impose $||\nabla f||=1$, but when dealing with residual streams $f(x)= x + g(x)$ you might end up having an identity mapping ($g(x)$ close to 0). 

Spectral Normalization is computed by dividing the weight matrix $W$ by the largest singular value. This value is computed using the power method, which is a cheap iterative method that approximates this value. The current way of imposing spectral normalization in pytorch is using network parametrization: `torch.nn.utils.parametrizations.spectral_norm`. In https://github.com/jhjacobsen/invertible-resnet * they propose an efficient method for imposing spectral normalization to convolutional layers. I propose an updated version using parametrizations.

*Jens Behrmann, Will Grathwohl, Ricky T. Q. Chen, David Duvenaud, Jörn-Henrik Jacobsen. Invertible Residual Networks. International Conference on Machine Learning (ICML), 2019. (https://icml.cc/)
