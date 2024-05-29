---
layout: post
read_time: true
show_date: true
title: Rutas Turísticas Recomendadas en la Toscana
date:   2024-05-12 13:32:20 -0600
description: Descubre los encantos de Roma. Explora los principales puntos turísticos, restaurantes locales y alojamientos en la Ciudad Eterna.
img: posts/20210312/nnet_optimization.jpg
tags: [coding, machine learning, optimization, deep Neural networks]
author: Javier Coco Gómez
github: amaynez/TicTacToe/blob/7bf83b3d5c10adccbeb11bf244fe0af8d9d7b036/entities/Neural_Network.py#L199
mathjax: yes # leave empty or erase to prevent the mathjax javascript from loading
toc: yes # leave empty or erase for no TOC
---
Bienvenidos a nuestra guía de viaje completa sobre Roma, la Ciudad Eterna. Roma, con su historia milenaria y su cultura vibrante, es un destino imprescindible para cualquier viajero. En esta guía, te llevaremos a través de los principales puntos turísticos, restaurantes locales y alojamientos asequibles para que disfrutes al máximo de tu visita.

## Día 1: Explora el Coliseo y el Foro Romano

Comienza tu aventura en Roma visitando el icónico Coliseo. Este anfiteatro antiguo es uno de los monumentos más famosos del mundo y ofrece una visión fascinante de la historia romana. Después, dirígete al Foro Romano, el corazón de la antigua Roma, donde podrás explorar ruinas y templos históricos.

## Día 2: Visita la Ciudad del Vaticano

Dedica tu segundo día a la Ciudad del Vaticano. Comienza con una visita a la Basílica de San Pedro y no te pierdas la impresionante vista desde la cúpula. Luego, explora los Museos Vaticanos, hogar de la Capilla Sixtina y otras obras maestras del Renacimiento.

## Día 3: Recorre el Centro Histórico

Pasea por el centro histórico de Roma y descubre la Fontana di Trevi, el Panteón y la Piazza Navona. Cada rincón de esta zona está lleno de historia y encanto. Asegúrate de lanzar una moneda en la Fontana di Trevi para asegurar tu regreso a Roma.

## Día 4: Descubre Trastevere

Explora el encantador barrio de Trastevere, conocido por sus calles adoquinadas y vibrante vida nocturna. Visita la Basílica de Santa María en Trastevere y disfruta de una cena en uno de los muchos restaurantes locales que ofrecen auténtica comida italiana.

## Día 5: Relájate en los Jardines de Villa Borghese

Concluye tu viaje relajándote en los Jardines de Villa Borghese. Este extenso parque ofrece hermosos paisajes, museos y vistas panorámicas de la ciudad desde la Terraza del Pincio. Puedes alquilar una bicicleta o simplemente disfrutar de un paseo tranquilo.

### Recomendaciones de Restaurantes

-La Carbonara: Prueba la mejor pasta carbonara en este restaurante histórico situado en el barrio de Monti.
-Pizzeria da Baffetto: Conocida por sus deliciosas pizzas, esta pizzería es una parada obligada para los amantes de la comida italiana.
-Roscioli: Disfruta de una experiencia gastronómica completa con una amplia selección de embutidos, quesos y vinos.

### Alojamiento Asequible

-Hotel Santa Maria: Un encantador hotel en Trastevere con cómodas habitaciones y un ambiente acogedor.
-Hotel Artimide: Situado cerca de la estación de Termini, ofrece habitaciones modernas y un excelente servicio al cliente.
-Generator Rome: Un albergue moderno y económico con una ubicación céntrica, ideal para mochileros y viajeros jóvenes.

Roma es una ciudad que te dejará sin aliento con su rica historia, impresionante arquitectura y deliciosa comida. Esperamos que esta guía de viaje completa te ayude a planificar tu visita y a disfrutar de todo lo que Roma tiene para ofrecer.


### Adam
[source](https://ruder.io/optimizing-gradient-descent/index.html#adam)

<p>Adaptive Moment Estimation (Adam) is an optimization method that computes adaptive learning rates for each weight and bias. In addition to storing an exponentially decaying average of past squared gradients \(v_t\) and an exponentially decaying average of past gradients \(m_t\), similar to momentum. Whereas momentum can be seen as a ball running down a slope, Adam behaves like a heavy ball with friction, which thus prefers flat minima in the error surface. We compute the decaying averages of past and past squared gradients \(m_t\) and \(v_t\) respectively as follows:</p>
<p style="text-align:center">\(<br>
\begin{align}<br>
\begin{split}<br>
m_t &amp;= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\<br>
v_t &amp;= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2<br>
\end{split}<br>
\end{align}<br>
\)</p>
<p>\(m_t\) and \(v_t\) are estimates of the first moment (the mean) and the second moment (the uncentered variance) of the gradients respectively, hence the name of the method. As \(m_t\) and \(v_t\) are initialized as vectors of 0's, the authors of Adam observe that they are biased towards zero, especially during the initial time steps, and especially when the decay rates are small (i.e. \(\beta_1\) and \(\beta_2\) are close to 1).</p>
<p>They counteract these biases by computing bias-corrected first and second moment estimates:</p>
<p style="text-align:center">\(<br>
\begin{align}<br>
\begin{split}<br>
\hat{m}_t &amp;= \dfrac{m_t}{1 - \beta^t_1} \\<br>
\hat{v}_t &amp;= \dfrac{v_t}{1 - \beta^t_2} \end{split}<br>
\end{align}<br>
\)</p>
<p>We then use these to update the weights and biases which yields the Adam update rule:</p>
<p style="text-align:center">\(\theta_{t+1} = \theta_{t} - \dfrac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t\).</p>
<p>The authors propose defaults of 0.9 for \(\beta_1\), 0.999 for \(\beta_2\), and \(10^{-8}\) for \(\epsilon\).</p>
[view on github](https://github.com/amaynez/TicTacToe/blob/b429e5637fe5f61e997f04c01422ad0342565640/entities/Neural_Network.py#L243)

```python
# decaying averages of past gradients
self.v["dW" + str(i)] = ((c.BETA1
                        * self.v["dW" + str(i)])
                        + ((1 - c.BETA1)
                        * np.array(self.gradients[i])
                        ))
self.v["db" + str(i)] = ((c.BETA1
                        * self.v["db" + str(i)])
                        + ((1 - c.BETA1)
                        * np.array(self.bias_gradients[i])
                        ))

# decaying averages of past squared gradients
self.s["dW" + str(i)] = ((c.BETA2
                        * self.s["dW"+str(i)])
                        + ((1 - c.BETA2)
                        * (np.square(np.array(self.gradients[i])))
                         ))
self.s["db" + str(i)] = ((c.BETA2
                        * self.s["db" + str(i)])
                        + ((1 - c.BETA2)
                        * (np.square(np.array(
                                         self.bias_gradients[i])))
                         ))

if c.ADAM_BIAS_Correction:
    # bias-corrected first and second moment estimates
    self.v["dW" + str(i)] = self.v["dW" + str(i)]
                          / (1 - (c.BETA1 ** true_epoch))
    self.v["db" + str(i)] = self.v["db" + str(i)]
                          / (1 - (c.BETA1 ** true_epoch))
    self.s["dW" + str(i)] = self.s["dW" + str(i)]
                          / (1 - (c.BETA2 ** true_epoch))
    self.s["db" + str(i)] = self.s["db" + str(i)]
                          / (1 - (c.BETA2 ** true_epoch))

# apply to weights and biases
weight_col -= ((eta * (self.v["dW" + str(i)]
                      / (np.sqrt(self.s["dW" + str(i)])
                      + c.EPSILON))))
self.bias[i] -= ((eta * (self.v["db" + str(i)]
                        / (np.sqrt(self.s["db" + str(i)])
                        + c.EPSILON))))
```


