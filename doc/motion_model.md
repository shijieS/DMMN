## Motion Model

- If  the motion is uniform motion, we recommend the motion model as follows:

  $f(t) = \frac{p_1t+p_2}{p_3t + 1}$

  for each variable of the bounding box.

- If the motion is accelerated motion, we recommend the motion model as follows:

  $f(t) = \frac{p_1t^2+p_2t+p_3}{p_4t^2+p_5t+1}$

  for each variable of the bounding box.

Here is the reason why we recommend this kind motion model:

|                                         Coordinate Transform |
| -----------------------------------------------------------: |
| $\left[\begin{matrix}X_C \\ Y_C \\ Z_C \\ 1\end{matrix}\right]=\left[\begin{matrix}r_{11} & r_{12} & r_{13} & T_x\\r_{21} & r_{32} & r_{43} & T_y \\r_{31} & r_{32} & r_{33} & T_z \\0 & 0 & 0 &1\end{matrix}\right] * \left[\begin{matrix}X_W \\ Y_W \\ Z_W \\ 1\end{matrix}\right]$ |
| $\left[\begin{matrix}x' \\ y' \\ Z_C\end{matrix}\right]=\left[\begin{matrix}f & 0 & 0 & 0\\0 & f & 0 & 0\\0 & 0 & 1 & \\\end{matrix}\right] *\left[\begin{matrix}r_{11} & r_{12} & r_{13} & T_x\\r_{21} & r_{32} & r_{43} & T_y \\r_{31} & r_{32} & r_{33} & T_z \\0 & 0 & 0 &1\end{matrix}\right] * \left[\begin{matrix}X_W \\ Y_W \\ Z_W \\ \end{matrix}\right]$ |
| $\left[\begin{matrix}u' \\ v' \\ Z_C \end{matrix}\right]=\left[\begin{matrix}\frac{1}{p_x} & 0 & -c_x\\0 & \frac{1}{p_y} & -c_y\\0 & 0 & 1\\\end{matrix}\right] *\left[\begin{matrix}f & 0 & 0 & 0\\0 & f & 0 & 0\\0 & 0 & 1 & 0\\\end{matrix}\right] *\left[\begin{matrix}r_{11} & r_{12} & r_{13} & T_x\\r_{21} & r_{32} & r_{43} & T_y \\r_{31} & r_{32} & r_{33} & T_z \\0 & 0 & 0 &1\end{matrix}\right] * \left[\begin{matrix}X_W \\ Y_W \\ Z_W \\ \end{matrix}\right]$ |

where, $\left[\begin{matrix}X_W \\ Y_W \\ Z_W \\ 1\end{matrix}\right]​$ is the point in world coordinate, $\left[\begin{matrix}X_C \\ Y_C \\ Z_C \\ 1\end{matrix}\right]​$ is the point in camera coordinate, $\left[\begin{matrix}x \\ y\end{matrix}\right]​$ is the point in image coordinate, $\left[\begin{matrix}u \\ v \end{matrix}\right]​$ is the point in the pixel coordinate and $\left\{
\begin{matrix}\left[\begin{matrix}x \\ y\end{matrix}\right] = 
\left[\begin{matrix}\frac{x'}{Z_C} \\ \frac{y'}{Z_C} \\ \end{matrix}\right] \\
\left[\begin{matrix}u \\ v \end{matrix}\right] = 
\left[\begin{matrix}\frac{u'}{X_C} \\ \frac{v'}{Z_C} \end{matrix}\right]\end{matrix}
\right.​$

If the motion is uniform motion, then $\left\{\begin{matrix}X_W = at+b \\ Y_W = ct+d \\ Z_W = mt+n\end{matrix}\right.​$ , where $a,b,c, d, m, n ​$ are the motion parameters in the world coordinate. By using the above formula, we can get:

$\left\{
\begin{matrix}
u = \frac{f}{p_x}\cdot\frac{
(r_{11}a + r_{12}c + r_{12}m)t + (r_{11}b + r_{12}d + r_{13}n + T_x)
}
{(r_{31}a + r_{32}c + r_{32}m)t + (r_{31}b + r_{32}d + r_{33}n + T_z)}
\\
v = \frac{f}{p_y}\cdot\frac{
(r_{21}a + r_{22}c + r_{22}m)t + (r_{21}b + r_{22}d + r_{23}n + T_y)
}
{(r_{31}a + r_{32}c + r_{32}m)t + (r_{31}b + r_{32}d + r_{33}n + T_z)}
\end{matrix}\right.$

Therefore, we can design  6 parameters to predict $u,v$, as follows

$\left\{\begin{matrix}
u = \frac{p_1t+p_2}{p_3t + p_4} \\
v = \frac{p_5t+p_6}{p_3t + p_4}
\end{matrix}\right.​$

For the simplicity, we introduce 2 parameters $(p_6, p_7)​$in order to make $u​$ independent from $v​$

$\left\{\begin{matrix}
u = \frac{p_1t+p_2}{p_3t + p_4} \\
v = \frac{p_5t+p_6}{p_7t + p_8}
\end{matrix}\right.$

Furthermore, We normalize these parameters:

$\left\{\begin{matrix}
u = \frac{p_1t+p_2}{p_3t + 1} \\
v = \frac{p_5t+p_6}{p_7t + 1}
\end{matrix}\right.​$

However, if I use the *scipy.optimize.curve_fit* directly, it will very hard to fit this curve. To solve this problems, we the following curve:

$\left\{\begin{matrix}
u'' = \mathop{log}(p_1t+p_2)-\mathop{log}(p_3t + 1) \\
v'' = \mathop{log}(p_5t+p_6)-\mathop{log}(p_7t + 1)
\end{matrix}\right.​$

Then, decode the $u, v​$

$\left\{\begin{matrix}u = e^{u''} \\ v = e^{v''}\end{matrix}\right.​$

Some range of parameters:

- In real scene, $t=0$, the points should be in the image, which means , the $p_3 \in [0, 1]$. 

- For each frame, the width and height of bounding boxes doesn't change. Therefore, for the parameters of width and height, let $\lim_{t\rightarrow \inf}\frac{p_1t+p_2}{p_3t + 1} = \frac{p_1}{p_3} \in [0, + \inf]​$,  

- IMPORTANT:  As to width and height, the derivative of this should be always be 0, So

  $(\frac{p_1t+p_2}{p_3t + 1})' = \frac{p_1(p_3t+1)-p_3(p_1t+p_2)}{(p_3t+1)^2}=0$

  So

  $p_1p_3t+p_1=p_1p_3t+p3 \\ p_1=p_3$

  