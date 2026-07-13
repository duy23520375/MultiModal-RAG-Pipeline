    Addressing Function Approximation Error in Actor-Critic Methods



        Scott Fujimoto 1 Herke van Hoof 2 David Meger 1

                           Abstract                                 means using an imprecise estimate within each update will
         In value-based reinforcement learning methods              lead to an accumulation of error. Due to overestimation bias,
        such as deep Q-learning, function approximation             this accumulated error can cause arbitrarily bad states to
        errors are known to lead to overestimated value             be estimated as high value, resulting in suboptimal policy
        estimates and suboptimal policies. We show that             updates and divergent behavior.
     this problem persists in an actor-critic setting and           This paper begins by establishing this overestimation prop-
       propose novel mechanisms to minimize its effects             erty is also present for deterministic policy gradients (Silver
        on both the actor and the critic. Our algorithm             et al., 2014), in the continuous control setting. Furthermore,
       builds on Double Q-learning, by taking the mini-             we find the ubiquitous solution in the discrete action setting,
      mum value between a pair of critics to limit over-            Double DQN (Van Hasselt et al., 2016), to be ineffective
        estimation. We draw the connection between tar-             in an actor-critic setting. During training, Double DQN
       get networks and overestimation bias, and suggest            estimates the value of the current policy with a separate tar-
      delaying policy updates to reduce per-update error            get value function, allowing actions to be evaluated without
         and further improve performance. We evaluate               maximization bias. Unfortunately, due to the slow-changing
         our method on the suite of OpenAI gym tasks,               policy in an actor-critic setting, the current and target value
       outperforming the state of the art in every envi-            estimates remain too similar to avoid maximization bias.
    ronment tested.                                                 This can be dealt with by adapting an older variant, Double
                                                                    Q-learning (Van Hasselt, 2010), to an actor-critic format
                                                                    by using a pair of independently trained critics. While this
    1. Introduction                                                 allows for a less biased value estimation, even an unbiased
                                                                    estimate with high variance can still lead to future overes-
    In reinforcement learning problems with discrete action         timations in local regions of state space, which in turn can
spaces, the issue of value overestimation as a result of func-      negatively affect the global policy. To address this concern,
  tion approximation errors is well-studied. However, similar       we propose a clipped Double Q-learning variant which lever-
  issues with actor-critic methods in continuous control do-        ages the notion that a value estimate suffering from overes-
   mains have been largely left untouched. In this paper, we        timation bias can be used as an approximate upper-bound to
   show overestimation bias and the accumulation of error in        the true value estimate. This favors underestimations, which
  temporal difference methods are present in an actor-critic        do not tend to be propagated during learning, as actions with
   setting. Our proposed method addresses these issues, and         low value estimates are avoided by the policy.
    greatly outperforms the current state of the art.
                                                                    Given the connection of noise to overestimation bias, this
 Overestimation bias is a property of Q-learning in which the       paper contains a number of components that address vari-
  maximization of a noisy value estimate induces a consistent       ance reduction. First, we show that target networks, a com-
    overestimation (Thrun & Schwartz, 1993). In a function          mon approach in deep Q-learning methods, are critical for
  approximation setting, this noise is unavoidable given the        variance reduction by reducing the accumulation of errors.
   imprecision of the estimator. This inaccuracy is further         Second, to address the coupling of value and policy, we
   exaggerated by the nature of temporal difference learning        propose delaying policy updates until the value estimate
  (Sutton, 1988), in which an estimate of the value function        has converged. Finally, we introduce a novel regularization
   is updated using the estimate of a subsequent state. This        strategy, where a SARSA-style update bootstraps similar
    1McGill University, Montreal, Canada 2University of Amster-     action estimates to further reduce variance.
dam, Amsterdam, Netherlands. Correspondence to: Scott Fujimoto      Our modifications are applied to the state of the art actor-
    <scott.fujimoto@mail.mcgill.ca>.                                critic method for continuous control, Deep Deterministic
 Proceedings of the 35 th International Conference on Machine       Policy Gradient algorithm (DDPG) (Lillicrap et al., 2015), to
  Learning, Stockholm, Sweden, PMLR 80, 2018. Copyright 2018        form the Twin Delayed Deep Deterministic policy gradient
    by the author(s).

    Addressing Function Approximation Error in Actor-Critic Methods

algorithm (TD3), an actor-critic algorithm which consid-          horizon. Another approach is a reduction in the discount
ers the interplay between function approximation error in         factor (Petrik & Scherrer, 2009), reducing the contribution
both policy and value updates. We evaluate our algorithm          of each error.
on seven continuous control domains from OpenAI gym               Our method builds on the Deterministic Policy Gradient
(Brockman et al., 2016), where we outperform the state of         algorithm (DPG) (Silver et al., 2014), an actor-critic method
the art by a wide margin.                                         which uses a learned value estimate to train a deterministic
Given the recent concerns in reproducibility (Henderson           policy. An extension of DPG to deep reinforcement learn-
et al., 2017), we run our experiments across a large num-         ing, DDPG (Lillicrap et al., 2015), has shown to produce
ber of seeds with fair evaluation metrics, perform abla-          state of the art results with an efficient number of iterations.
tion studies across each contribution, and open source both       Orthogonal to our approach, recent improvements to DDPG
our code and learning curves (https://github.com/                 include distributed methods (Popov et al., 2017), along with
sfujim/TD3).                                                      multi-step returns and prioritized experience replay (Schaul
                                                                  et al., 2016; Horgan et al., 2018), and distributional methods
2. Related Work                                                   (Bellemare et al., 2017; Barth-Maron et al., 2018).
Function approximation error and its effect on bias and           3. Background
variance in reinforcement learning algorithms have been
studied in prior works (Pendrith et al., 1997; Mannor et al.,     Reinforcement learning considers the paradigm of an agent
2007). Our work focuses on two outcomes that occur as the         interacting with its environment with the aim of learning
result of estimation error, namely overestimation bias and a      reward-maximizing behavior. At each discrete time step
high variance build-up.                                           t, with a given state s ∈ S, the agent selects actions
Several approaches exist to reduce the effects of overestima-     a ∈ A with respect to its policy π : S → A, receiv-
                                                                  ing a reward r and the new state of the environment s        0.
tion bias due to function approximation and policy optimiza-      The return is defined as the discounted sum of rewards
tion in Q-learning. Double Q-learning uses two independent            PT     i-t
estimators to make unbiased value estimates (Van Hasselt,         Rt =     i=t γ     r(si, ai), where γ is a discount factor de-
2010; Van Hasselt et al., 2016). Other approaches have            termining the priority of short-term rewards.
focused directly on reducing the variance (Anschel et al.,        In reinforcement learning, the objective is to find the op-
2017), minimizing over-fitting to early high variance esti-       timal policy πφ, with parameters φ, which maximizes the
mates (Fox et al., 2016), or through corrective terms (Lee        expected return J(φ) = Esi∼pπ,ai∼π [R0]. For continuous
et al., 2013). Further, the variance of the value estimate        control, parametrized policies πφ can be updated by taking
has been considered directly for risk-aversion (Mannor &
Tsitsiklis, 2011) and exploration (O'Donoghue et al., 2017),      the gradient of the expected return ∇φJ(φ). In actor-critic
                                                                  methods, the policy, known as the actor, can be updated
but without connection to overestimation bias.                    through the deterministic policy gradient algorithm (Silver
The concern of variance due to the accumulation of error in       et al., 2014):
temporal difference learning has been largely dealt with by       ∇φJ(φ) = Es∼pπ -∇aQπ(s, a)|a=π(s)∇φπφ(s) .                (1)
either minimizing the size of errors at each time step or mix-
ing off-policy and Monte-Carlo returns. Our work shows            Qπ(s, a)          = Esi∼pπ,ai∼π [Rt|s, a], the expected return
the importance of a standard technique, target networks, for      when performing action a in state s and following π af-
the reduction of per-update error, and develops a regulariza-     ter, is known as the critic or the value function.
tion technique for the variance reduction by averaging over       In Q-learning, the value function can be learned using tem-
value estimates. Concurrently, Nachum et al. (2018) showed        poral difference learning (Sutton, 1988; Watkins, 1989), an
smoothed value functions could be used to train stochastic        update rule based on the Bellman equation (Bellman, 1957).
policies with reduced variance and improved performance.          The Bellman equation is a fundamental relationship between
Methods with multi-step returns offer a trade-off between         the value of a state-action pair (s, a) and the value of the
accumulated estimation bias and variance induced by the           subsequent state-action pair (s0, a0):
policy and the environment. These methods have been               Qπ    (s, a) = r + γEs ,a0 [Q π(s0, a0)] ,  a0 ∼ π(s0
shown to be an effective approach, through importance sam-            0        ).                                            (2)
pling (Precup et al., 2001; Munos et al., 2016), distributed
methods (Mnih et al., 2016; Espeholt et al., 2018), and ap-       For a large state space, the value can be estimated with a
proximate bounds (He et al., 2016). However, rather than          differentiable function approximator Qθ(s, a), with param-
provide a direct solution to the accumulation of error, these     eters θ. In deep Q-learning (Mnih et al., 2015), the network
methods circumvent the problem by considering a longer            is updated by using temporal difference learning with a sec-
                                                                  ondary frozen target network Qθ0 (s, a) to maintain a fixed

                                          Addressing Function Approximation Error in Actor-Critic Methods

objective y over multiple updates:                                     400                               500
    0 (s    0, a0    ),      a0 ∼ πφ0 (s0                              300                               400
          y = r + γQθ        ),                                 (3)    Average Value200
                                                                                                         200
where the actions are selected from a target actor network             100    CDQ           True CDQ     100
πφ0.             The weights of a target network are either updated     0    DDPG           True DDPG      0
periodically to exactly match the weights of the current            0.0 0.2 0.4 0.6 0.8 1.0                     0.0 0.2 0.4 0.6 0.8 1.0
                                                                       Time steps (1e6)                     Time steps (1e6)
                                                                0 ←      (a) Hopper-v1
network, or by some proportion τ at each time step θ                                                        (b) Walker2d-v1
τθ + (1 - τ )θ0       . This update can be applied in an off-policy
fashion, sampling random mini-batches of transitions from               Figure 1. Measuring overestimation bias in the value estimates
an experience replay buffer (Lin, 1992).                                of DDPG and our proposed method, Clipped Double Q-learning
                                                                        (CDQ), on MuJoCo environments over 1 million time steps.
4. Overestimation Bias
In Q-learning with discrete actions, the value estimate is              occur with slightly stricter conditions. We examine this case
updated with a greedy target y = r + γ maxa0 Q(s0, a0),                 further in the supplementary material. We denote πapprox
however, if the target is susceptible to error , then the max-         and πtrue as the policy with parameters φapprox and φtrue re-
imum over the value along with its error will generally be              spectively.
greater than the true maximum, E[maxa0 (Q(s0, a0           )+)] ≥     As the gradient direction is a local maximizer, there exists 1
maxa0 Q(s0, a0) (Thrun & Schwartz, 1993). As a result,
even initially zero-mean error can cause value updates to               sufficiently small such that if α ≤ 1 then the approximate
                                                                        value of πapprox will be bounded below by the approximate
result in a consistent overestimation bias, which is then prop-         value of πtrue:
agated through the Bellman equation. This is problematic as
errors induced by function approximation are unavoidable.    E [Qθ(s, πapprox(s))] ≥ E [Qθ(s, πtrue(s))] .                          (5)
While in the discrete action setting overestimation bias is             Conversely, there exists 2 sufficiently small such that if
an obvious artifact from the analytical maximization, the
presence and effects of overestimation bias is less clear in an         α ≤ 2 then the true value of πapprox will be bounded above
                                                                        by the true value of πtrue:
actor-critic setting where the policy is updated via gradient
descent. We begin by proving that the value estimate in de-                 E [Qπ    (s, πtrue(s))] ≥ E [Q π(s, πapprox(s))] .      (6)
terministic policy gradients will be an overestimation under            If in expectation the value estimate is at least as large as
some basic assumptions in Section 4.1 and then propose
a clipped variant of Double Q-learning in an actor-critic               the true value with respect to φtrue, E [Qθ (s, πtrue(s))] ≥
setting to reduce overestimation bias in Section 4.2.                   E [Qπ (s, πtrue(s))], then Equations (5) and (6) imply that if
                                                                        α < min(1, 2), then the value estimate will be overesti-
4.1. Overestimation Bias in Actor-Critic                                mated:
                                                                                                            π
In actor-critic methods the policy is updated with respect                  E [Qθ(s, πapprox(s))] ≥ E [Q     (s, πapprox(s))] .     (7)
to the value estimates of an approximate critic. In this
section we assume the policy is updated using the deter-                Although this overestimation may be minimal with each
ministic policy gradient, and show that the update induces              update, the presence of error raises two concerns. Firstly, the
overestimation in the value estimate. Given current policy              overestimation may develop into a more significant bias over
parameters φ, let φapprox define the parameters from the ac-            many updates if left unchecked. Secondly, an inaccurate
tor update induced by the maximization of the approximate               value estimate may lead to poor policy updates. This is
critic Qθ(s, a) and φtrue the parameters from the hypothet-             particularly problematic because a feedback loop is created,
ical actor update with respect to the true underlying value             in which suboptimal actions might be highly rated by the
function Qπ(s, a) (which is not known during learning):                 suboptimal critic, reinforcing the suboptimal action in the
               α                                                        next policy update.
φapprox = φ + Z1    Es∼pπ -∇φπφ(s)∇aQθ(s, a)|a=πφ(s)                   Does this theoretical overestimation occur in practice
               α        π                         (s, a)|a=πφ(s) ,     for state-of-the-art methods? We answer this question by
  φtrue = φ + Z2    Es∼pπ -∇φπφ(s)∇aQ                                   plotting the value estimate of DDPG (Lillicrap et al., 2015)
                                                                (4)     over time while it learns on the OpenAI gym environments
where we assume Z1 and Z2 are chosen to normalize the                   Hopper-v1 and Walker2d-v1 (Brockman et al., 2016). In
gradient, i.e., such that Z-1         ||E[·]|| = 1. Without normal-     Figure 1, we graph the average value estimate over 10000
ized gradients, overestimation bias is still guaranteed to              states and compare it to an estimate of the true value. The

                            Addressing Function Approximation Error in Actor-Critic Methods

    400                                   400                                    demonstrates that the actor-critic Double DQN suffers from
    300                                   300                                    a similar overestimation as DDPG (as shown in Figure 1).
    200                                                                          While Double Q-learning is more effective, it does not en-
                                          200                                    tirely eliminate the overestimation. We further show this
    100       DQ-AC     True DQ-AC        100                                    reduction is not sufficient experimentally in Section 6.1.
      0       DDQN-AC     True DDQN-AC      0
               0.0 0.2 0.4 0.6 0.8 1.0           0.0 0.2 0.4 0.6 0.8 1.0         As πφ1 optimizes with respect to Qθ1 , using an indepen-
Time steps (1e6)    Time steps (1e6)
                     (b) Walker2d-v1                                             bias introduced by the policy update. However the critics
              (a) Hopper-v1                                                      dent estimate in the target update of Qθ1 would avoid the
    Figure 2. Measuring overestimation bias in the value estimates of            are not entirely independent, due to the use of the oppo-
    actor critic variants of Double DQN (DDQN-AC) and Double Q-                  site critic in the learning targets, as well as the same re-
    learning (DQ-AC) on MuJoCo environments over 1 million time                  play buffer. As a result, for some states s we will have
    steps.                                                                       Qθ2 (s, πφ1(s)) > Qθ1(s, πφ1(s)). This is problematic be-
                                                                                 cause Qθ1(s, πφ1(s)) will generally overestimate the true
                                                                                 value, and in certain areas of the state space the overestima-
    true value is estimated using the average discounted return                  tion will be further exaggerated. To address this problem,
    over 1000 episodes following the current policy, starting                    we propose to simply upper-bound the less biased value
    from states sampled from the replay buffer. A very clear                     estimate Qθ2 by the biased estimate Qθ1 .      This results in
    overestimation bias occurs from the learning procedure,                      taking the minimum between the two estimates, to give the
    which contrasts with the novel method that we describe in                    target update of our Clipped Double Q-learning algorithm:
    the following section, Clipped Double Q-learning, which
    greatly reduces overestimation by the critic.                                y1 = r + γ min Qθ         0(s0, πφ1 (s    0 )).           (10)
                                                                                 i=1,2                     i

    4.2. Clipped Double Q-Learning for Actor-Critic                              With Clipped Double Q-learning, the value target cannot
    While several approaches to reducing overestimation bias                     introduce any additional overestimation over using the stan-
    have been proposed, we find them ineffective in an actor-                    dard Q-learning target. While this update rule may induce
    critic setting. This section introduces a novel clipped variant              an underestimation bias, this is far preferable to overesti-
    of Double Q-learning (Van Hasselt, 2010), which can re-                      mation bias, as unlike overestimated actions, the value of
    place the critic in any actor-critic method.                                 underestimated actions will not be explicitly propagated
                                                                                 through the policy update.
    In Double Q-learning, the greedy update is disentangled                      In implementation, computational costs can be reduced by
    from the value function by maintaining two separate value                    using a single actor optimized with respect to Qθ1 . We then
    estimates, each of which is used to update the other. If the                 use the same target y2 = y1 for Qθ2 . If Qθ2 > Qθ1 then
    value estimates are independent, they can be used to make                    the update is identical to the standard update and induces no
    unbiased estimates of the actions selected using the opposite                additional bias. If Qθ2 < Qθ1, this suggests overestimation
    value estimate. In Double DQN (Van Hasselt et al., 2016),                    has occurred and the value is reduced similar to Double Q-
    the authors propose using the target network as one of the                   learning. A proof of convergence in the finite MDP setting
    value estimates, and obtain a policy by greedy maximization                  follows from this intuition. We provide formal details and
    of the current value network rather than the target network.                 justification in the supplementary material.
    In an actor-critic setting, an analogous update uses the cur-
    rent policy rather than the target policy in the learning target:            A secondary benefit is that by treating the function approxi-
                                                                                 mation error as a random variable we can see that the min-
                              y = r + γQθ0 (s0   , πφ(s   0)).           (8)     imum operator should provide higher value to states with
    In practice however, we found that with the slow-changing                    lower variance estimation error, as the expected minimum
    policy in actor-critic, the current and target networks were                 of a set of random variables decreases as the variance of
    too similar to make an independent estimation, and offered                   the random variables increases. This effect means that the
    little improvement. Instead, the original Double Q-learning                  minimization in Equation (10) will lead to a preference for
    formulation can be used, with a pair of actors (πφ1, πφ2)                    states with low-variance value estimates, leading to safer
    and critics (Qθ1 , Qθ2 ), where πφ1 is optimized with respect                policy updates with stable learning targets.
    to Qθ1 and πφ2 with respect to Qθ2 :
                                           (s                                    5. Addressing Variance
    0                     y1 = r + γQθ           0        (s0   ))
                                      2          0, πφ1    0             (9)     While Section 4 deals with the contribution of variance to
    0                     y2 = r + γQθ                                           overestimation bias, we also argue that variance itself should
                                      1   (s , πφ2 (s )).

    We measure the overestimation bias in Figure 2, which                        be directly addressed. Besides the impact on overestimation










Average Value

                                                      Addressing Function Approximation Error in Actor-Critic Methods

bias, high variance estimates provide a noisy gradient for the                                                           104
policy update. This is known to reduce learning speed (Sut-                          350
ton & Barto, 1998) as well as hurt performance in practice.            Average Value 300                                 103
In this section we emphasize the importance of minimizing                            250
error at each update, build the connection between target                            200     τ = 1       τ = 0.01
networks and estimation error and propose modifications to                           150     τ = 0.1     True Value      102
the learning procedure of actor-critic for variance reduction.                                0.0 0.2 0.4 0.6 0.8 1.0       0.0 0.2 0.4 0.6 0.8 1.0
                                                                                         Time steps (1e5)    Time steps (1e5)
                                                                                         (a) Fixed Policy   (b) Learned Policy
5.1. Accumulating Error
                                                                       Figure 3. Average estimated value of a randomly selected state
Due to the temporal difference update, where an estimate of            on Hopper-v1 without target networks, (τ = 1), and with slow-
the value function is built from an estimate of a subsequent           updating target networks, (τ = 0.1, 0.01), with a fixed and a
state, there is a build up of error. While it is reasonable to         learned policy.
expect small error for an individual update, these estimation
errors can accumulate, resulting in the potential for large
overestimation bias and suboptimal policy updates. This is
exacerbated in a function approximation setting where the              procedure, and allow a greater coverage of the training data.
Bellman equation is never exactly satisfied, and each update           Without a fixed target, each update may leave residual error
leaves some amount of residual TD-error δ(s, a):                       which will begin to accumulate. While the accumulation of
                                                                       error can be detrimental in itself, when paired with a policy
Qθ(s, a) = r + γE[Qθ(s    0, a0    )] - δ(s, a).              (11)     maximizing over the value estimate, it can result in wildly
                                                                       divergent values.
It can then be shown that rather than learning an estimate             To provide some intuition, we examine the learning behavior
of the expected return, the value estimate approximates the            with and without target networks on both the critic and actor
expected return minus the expected discounted sum of future            in Figure 3, where we graph the value, in a similar manner to
TD-errors:                                                             Figure 1, in the Hopper-v1 environment. In (a) we compare
                                                                       the behavior with a fixed policy and in (b) we examine the
Qθ(st, at) = rt + γE[Qθ(st+1, at+1)] - δt                              value estimates with a policy that continues to learn, trained
= rt + γE [rt+1 + γE [Qθ(st+2, at+2) - δt+1]] - δt                     with the current value estimate. The target networks use a
                         "X        #                                   slow-moving update rate, parametrized by τ
= Esi∼pπ,ai∼π             T    γi-t(ri - δi)  .               (12)                                                         .
                        i=t                                            While updating the value estimate without target networks
                                                                       (τ = 1) increases the volatility, all update rates result in sim-
If the value estimate is a function of future reward and es-           ilar convergent behaviors when considering a fixed policy.
timation error, it follows that the variance of the estimate           However, when the policy is trained with the current value
will be proportional to the variance of future reward and es-          estimate, the use of fast-updating target networks results in
timation error. Given a large discount factor γ, the variance          highly divergent behavior.
can grow rapidly with each update if the error from each
update is not tamed. Furthermore each gradient update only             When do actor-critic methods fail to learn? These results
reduces error with respect to a small mini-batch which gives           suggest that the divergence that occurs without target net-
no guarantees about the size of errors in value estimates              works is the result of policy updates with a high variance
outside the mini-batch.                                                value estimate. Figure 3, as well as Section 4, suggest failure
                                                                       can occur due to the interplay between the actor and critic
5.2. Target Networks and Delayed Policy Updates                        updates. Value estimates diverge through overestimation
                                                                       when the policy is poor, and the policy will become poor if
In this section we examine the relationship between target             the value estimate itself is inaccurate.
networks and function approximation error, and show the                If target networks can be used to reduce the error over mul-
use of a stable target reduces the growth of error. This               tiple updates, and policy updates on high-error states cause
insight allows us to consider the interplay between high               divergent behavior, then the policy network should be up-
variance estimates and policy performance, when designing              dated at a lower frequency than the value network, to first
reinforcement learning algorithms.                                     minimize error before introducing a policy update. We pro-
Target networks are a well-known tool to achieve stabil-               pose delaying policy updates until the value error is as small
ity in deep reinforcement learning. As deep function ap-               as possible. The modification is to only update the policy
proximators require multiple gradient updates to converge,             and target networks after a fixed number of updates d to the
target networks provide a stable objective in the learning             critic. To ensure the TD-error remains small, we update the

    Addressing Function Approximation Error in Actor-Critic Methods

target networks slowly θ   0 ← τθ + (1 - τ )θ0.        Algorithm 1 TD3
By sufficiently delaying the policy updates we limit the like-          Initialize critic networks Qθ1 , Qθ2, and actor network πφ
lihood of repeating updates with respect to an unchanged                with random parameters θ1, θ2, φ
critic. The less frequent policy updates that do occur will             Initialize target networks θ 0    0               0 ← φ
use a value estimate with lower variance, and in principle,                                          1 ← θ1, θ2 ← θ2, φ
should result in higher quality policy updates. This creates a          for t = 1 to T do
two-timescale algorithm, as often required for convergence             Select action with exploration noise a ∼ π(s) + ,    0
in the linear setting (Konda & Tsitsiklis, 2003). The effec-
tiveness of this strategy is captured by our empirical results         Store transition tuple (s, a, r, s0
presented in Section 6.1, which show an improvement in                                                        ) in B
performance while using fewer policy updates.                          Sample mini-batch of N transitions (s, a, r, s0) from B
                                                                       a˜ ← πφ0 (s) + ,   ∼ clip(N (0, σ˜), -c, c)
                                                                                                              0, a˜)
                                                                       y ← r + γ mini=1,2 Qθ (s
5.3. Target Policy Smoothing Regularization                                                         0
                                                                                                    i
                                                                       Update critics θi ← minθi N -1 P(y - Qθi(s, a))2
A concern with deterministic policies is they can overfit              if t mod d then
to narrow peaks in the value estimate. When updating the               Update φ by the deterministic policy gradient:
critic, a learning target using a deterministic policy is highly
susceptible to inaccuracies induced by function approxima-             ∇φJ(φ) = N -1 P∇aQθ1(s, a)|a=πφ(s)∇φπφ(s)
                                                                       Update target networks:
tion error, increasing the variance of the target. This induced        θ       0                    0
                                                                                 i ← τθi + (1 - τ )θi
variance can be reduced through regularization. We intro-              φ
                                                                                 0 ← τφ + (1 - τ )φ0
duce a regularization strategy for deep value learning, target         end if
policy smoothing, which mimics the learning update from                 end for
SARSA (Sutton & Barto, 1998). Our approach enforces
the notion that similar actions should have similar value.
While the function approximation does this implicitly, the
relationship between similar actions can be forced explicitly
by modifying the training procedure. We propose that fitting
the value of a small area around the target action
         y = r + E [Qθ0 (s0, πφ0 (s0) + )] ,                 (13)
would have the benefit of smoothing the value estimate by               (a)        (b)                        (c)       (d)
bootstrapping off of similar state-action value estimates. In           Figure 4. Example MuJoCo environments (a) HalfCheetah-v1, (b)
practice, we can approximate this expectation over actions              Hopper-v1, (c) Walker2d-v1, (d) Ant-v1.
by adding a small amount of random noise to the target
policy and averaging over mini-batches. This makes our
modified target update:                                                 6. Experiments
            y = r + γQθ0 (s0, πφ0 (s0) + ),                   (14)     We present the Twin Delayed Deep Deterministic policy
     ∼ clip(N (0, σ), -c, c),                                          gradient algorithm (TD3), which builds on the Deep Deter-
                                                                        ministic Policy Gradient algorithm (DDPG) (Lillicrap et al.,
where the added noise is clipped to keep the target in a                2015) by applying the modifications described in Sections
small range. The outcome is an algorithm reminiscent of                 4.2, 5.2 and 5.3 to increase the stability and performance
Expected SARSA (Van Seijen et al., 2009), where the value               with consideration of function approximation error. TD3
estimate is instead learned off-policy and the noise added to           maintains a pair of critics along with a single actor. For each
the target policy is chosen independently of the exploration            time step, we update the pair of critics towards the minimum
policy. The value estimate learned is with respect to a noisy           target value of actions selected by the target policy:
policy defined by the parameter σ.
Intuitively, it is known that policies derived from SARSA                          y = r + γ min Qθ       0(s 0, πφ0 (s0) + ),
value estimates tend to be safer, as they provide higher value                     i=1,2                  i                        (15)
to actions resistant to perturbations. Thus, this style of                          ∼ clip(N (0, σ), -c, c).
update can additionally lead to improvement in stochastic
domains with failure cases. A similar idea was introduced               Every d iterations, the policy is updated with respect to Qθ1
concurrently by Nachum et al. (2018), smoothing over Qθ,                following the deterministic policy gradient algorithm (Silver
rather than Qθ0.                                                        et al., 2014). TD3 is summarized in Algorithm 1.

                                      Addressing Function Approximation Error in Actor-Critic Methods

                                      TD3           DDPG     our DDPG                  PPO     TRPO     ACKTR             SAC
                10000                               3500                               5000
                8000                                3000                               4000                            4000
                6000                                2500                               3000                            3000
                                                    2000                                                               2000
                4000                                1500                               2000
                2000                                1000                                                               1000
                                                    500                                1000                               0
                        00.0 0.2 0.4 0.6 0.8 1.0           00.0 0.2 0.4 0.6 0.8 1.0           00.0 0.2 0.4 0.6 0.8 1.0-10000.0 0.2 0.4 0.6 0.8 1.0
                            Time steps (1e6)                   Time steps (1e6)                   Time steps (1e6)            Time steps (1e6)
Average Return      Average Return
                    (a) HalfCheetah-v1                           (b) Hopper-v1                     (c) Walker2d-v1        (d) Ant-v1
                                   -4                                1000                               10000
                                                                      900                               8000
                                   -6                                 800                               6000
                                   -8                                 700
                                                                      600                               4000
                                   -10                                500                               2000
                                   -120.0 0.2 0.4 0.6 0.8 1.0         4000.0 0.2 0.4 0.6 0.8 1.0            00.0 0.2 0.4 0.6 0.8 1.0
                                            Time steps (1e6)                Time steps (1e6)                    Time steps (1e6)
                                              (e) Reacher-v1             (f) InvertedPendulum-v1          (g) InvertedDoublePendulum-v1

Figure 5. Learning curves for the OpenAI gym continuous control tasks. The shaded region represents half a standard deviation of the
average evaluation over 10 trials. Some graphs are cropped to display the interesting regions.


Table 1. Max Average Return over 10 trials of 1 million time steps. Maximum value for each task is bolded. ± corresponds to a single
standard deviation over trials.

Environment              TD3            DDPG     Our DDPG     PPO                                    TRPO     ACKTR       SAC
HalfCheetah       9636.95 ± 859.065    3305.60    8577.29   1795.43                                -15.57    1450.46    2347.19
Hopper            3564.07 ± 114.74     2020.46    1860.02   2164.70                               2471.30    2428.39    2996.66
Walker2d          4682.82 ± 539.64     1843.85    3098.11   3317.69                               2321.47    1216.70    1283.67
Ant               4372.44 ± 1000.33    1005.30    888.77    1083.20                                -75.85    1821.94    655.35
Reacher             -3.60 ± 0.56        -6.51      -4.01     -6.18                                -111.43     -4.26      -4.44
InvPendulum        1000.00 ± 0.00      1000.00    1000.00   1000.00                                985.40    1000.00    1000.00
InvDoublePendulum  9337.47 ± 14.96     9355.52    8369.95   8977.94                                205.85    9081.92    8487.15


6.1. Evaluation                                                             N (0, 0.2) to the actions chosen by the target actor network,
To evaluate our algorithm, we measure its performance on                    clipped to (-0.5, 0.5), delayed policy updates consists of
the suite of MuJoCo continuous control tasks (Todorov et al.,               only updating the actor and target critic network every d
2012), interfaced through OpenAI Gym (Brockman et al.,                      iterations, with d = 2. While a larger d would result in a
2016) (Figure 4). To allow for reproducible comparison, we                  larger benefit with respect to accumulating errors, for fair
use the original set of tasks from Brockman et al. (2016)                   comparison, the critics are only trained once per time step,
with no modifications to the environment or reward.                         and training the actor for too few iterations would cripple
                                                                            learning. Both target networks are updated with τ = 0.005.
For our implementation of DDPG (Lillicrap et al., 2015), we                 To remove the dependency on the initial parameters of the
use a two layer feedforward neural network of 400 and 300                   policy we use a purely exploratory policy for the first 10000
hidden nodes respectively, with rectified linear units (ReLU)               time steps of stable length environments (HalfCheetah-v1
between each layer for both the actor and critic, and a final               and Ant-v1) and the first 1000 time steps for the remaining
tanh unit following the output of the actor. Unlike the orig-               environments. Afterwards, we use an off-policy exploration
inal DDPG, the critic receives both the state and action as
input to the first layer. Both network parameters are updated               strategy, adding Gaussian noise N (0, 0.1) to each action.
using Adam (Kingma & Ba, 2014) with a learning rate of                      Unlike the original implementation of DDPG, we used un-
10-3            . After each time step, the networks are trained with a     correlated noise for exploration as we found noise drawn
mini-batch of a 100 transitions, sampled uniformly from a                   from the Ornstein-Uhlenbeck (Uhlenbeck & Ornstein, 1930)
replay buffer containing the entire history of the agent.                   process offered no performance benefits.
                                                                            Each task is run for 1 million time steps with evaluations
The target policy smoothing is implemented by adding  ∼                    every 5000 time steps, where each evaluation reports the

    Addressing Function Approximation Error in Actor-Critic Methods

average reward over 10 episodes with no exploration noise.        Table 2. Average return over the last 10 evaluations over 10 trials
Our results are reported over 10 random seeds of the Gym          of 1 million time steps, comparing ablation over delayed policy
simulator and the network initialization.                         updates (DP), target policy smoothing (TPS), Clipped Double
We compare our algorithm against DDPG (Lillicrap et al.,          Q-learning (CDQ) and our architecture, hyper-parameters and
2015) as well as the state of art policy gradient algorithms:     exploration (AHE). Maximum value for each task is bolded.
PPO (Schulman et al., 2017), ACKTR (Wu et al., 2017)
and TRPO (Schulman et al., 2015), as implemented by                Method      HCheetah     Hopper    Walker2d    Ant
OpenAI's baselines repository (Dhariwal et al., 2017), and         TD3          9532.99     3304.75    4565.24            4185.06
SAC (Haarnoja et al., 2018), as implemented by the author's        DDPG         3162.50     1731.94    1520.90             816.35
GitHub1. Additionally, we compare our method with our              AHE          8401.02     1061.77    2362.13             564.07
re-tuned version of DDPG, which includes all architecture          AHE + DP     7588.64     1465.11    2459.53             896.13
and hyper-parameter modifications to DDPG without any              AHE + TPS    9023.40     907.56     2961.36             872.17
of our proposed adjustments. A full comparison between             AHE + CDQ    6470.20     1134.14    3979.21            3818.71
our re-tuned version and the baselines DDPG is provided in         TD3 - DP     9590.65     2407.42    4695.50            3754.26
the supplementary material.                                        TD3 - TPS    8987.69     2392.59    4033.67            4155.24
                                                                   TD3 - CDQ    9792.80     1837.32    2579.39             849.75
Our results are presented in Table 1 and learning curves in        DQ-AC
Figure 5. TD3 matches or outperforms all other algorithms                       9433.87     1773.71    3100.45            2445.97
in both final performance and learning speed across all tasks.     DDQN-AC     10306.90     2155.75    3116.81            1092.18

6.2. Ablation Studies                                             outperforms both prior methods, this suggests that subdu-
We perform ablation studies to understand the contribution        ing the overestimations from the unbiased estimator is an
of each individual component: Clipped Double Q-learning           effective measure to improve performance.
(Section 4.2), delayed policy updates (Section 5.2) and target
policy smoothing (Section 5.3). We present our results in         7. Conclusion
Table 2 in which we compare the performance of removing
each component from TD3 along with our modifications to           Overestimation has been identified as a key problem in
the architecture and hyper-parameters. Additional learning        value-based methods. In this paper, we establish overesti-
curves can be found in the supplementary material.                mation bias is also problematic in actor-critic methods. We
                                                                  find the common solutions for reducing overestimation bias
The significance of each component varies task to task.           in deep Q-learning with discrete actions are ineffective in an
While the addition of only a single component causes in-          actor-critic setting, and develop a novel variant of Double
significant improvement in most cases, the addition of com-       Q-learning which limits possible overestimation. Our re-
binations performs at a much higher level. The full algo-         sults demonstrate that mitigating overestimation can greatly
rithm outperforms every other combination in most tasks.          improve the performance of modern algorithms.
Although the actor is trained for only half the number of
iterations, the inclusion of delayed policy update generally      Due to the connection between noise and overestimation,
improves performance, while reducing training time.               we examine the accumulation of errors from temporal dif-
                                                                  ference learning. Our work investigates the importance of
We additionally compare the effectiveness of the actor-critic     a standard technique in deep reinforcement learning, target
variants of Double Q-learning (Van Hasselt, 2010) and Dou-        networks, and examines their role in limiting errors from
ble DQN (Van Hasselt et al., 2016), denoted DQ-AC and             imprecise function approximation and stochastic optimiza-
DDQN-AC respectively, in Table 2. For fairness in com-            tion. Finally, we introduce a SARSA-style regularization
parison, these methods also benefited from delayed policy         technique which modifies the temporal difference target to
updates, target policy smoothing and use our architecture         bootstrap off similar state-action pairs.
and hyper-parameters. Both methods were shown to reduce
overestimation bias less than Clipped Double Q-learning in        Taken together, these improvements define our proposed
Section 4. This is reflected empirically, as both methods         approach, the Twin Delayed Deep Deterministic policy gra-
result in insignificant improvements over TD3 - CDQ, with         dient algorithm (TD3), which greatly improves both the
an exception in the Ant-v1 environment, which appears to          learning speed and performance of DDPG in a number of
benefit greatly from any overestimation reduction. As the         challenging tasks in the continuous control setting. Our
inclusion of Clipped Double Q-learning into our full method       algorithm exceeds the performance of numerous state of
                                                                  the art algorithms. As our modifications are simple to im-
1See the supplementary material for hyper-parameters and a        plement, they can be easily added to any other actor-critic
discussion on the discrepancy in the reported results of SAC.     algorithm.

    Addressing Function Approximation Error in Actor-Critic Methods

References                                                                    Kingma, D. and Ba, J. Adam: A method for stochastic
          Anschel, O., Baram, N., and Shimkin, N. Averaged-dqn:     optimization. arXiv preprint arXiv:1412.6980, 2014.
Variance reduction and stabilization for deep reinforce-           Konda, V. R. and Tsitsiklis, J. N. On actor-critic algorithms.
ment learning. In International Conference on Machine               SIAM journal on Control and Optimization, 42(4):1143-
Learning, pp. 176-185, 2017.                                        1166, 2003.

           Barth-Maron, G., Hoffman, M. W., Budden, D., Dabney,           Lee, D., Defourny, B., and Powell, W. B. Bias-corrected
W., Horgan, D., TB, D., Muldal, A., Heess, N., and Lil-             q-learning to control max-operator bias in q-learning.
licrap, T. Distributional policy gradients. International           In Adaptive Dynamic Programming And Reinforcement
Conference on Learning Representations, 2018.                       Learning (ADPRL), 2013 IEEE Symposium on, pp. 93-99.
                                                                    IEEE, 2013.
        Bellemare, M. G., Dabney, W., and Munos, R. A distribu-
tional perspective on reinforcement learning. In Interna-            Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez,
tional Conference on Machine Learning, pp. 449-458,                 T., Tassa, Y., Silver, D., and Wierstra, D. Continuous
2017.                                                               control with deep reinforcement learning. arXiv preprint
                                                                    arXiv:1509.02971, 2015.
          Bellman, R. Dynamic Programming. Princeton University     Lin, L.-J. Self-improving reactive agents based on reinforce-
Press, 1957.                                                        ment learning, planning and teaching. Machine learning,
       Brockman, G., Cheung, V., Pettersson, L., Schneider, J.,     8(3-4):293-321, 1992.
Schulman, J., Tang, J., and Zaremba, W. Openai gym,                   Mannor, S. and Tsitsiklis, J. N. Mean-variance optimization
2016.                                                               in markov decision processes. In International Confer-
     Dhariwal, P., Hesse, C., Plappert, M., Radford, A., Schul-     ence on Machine Learning, pp. 177-184, 2011.
man, J., Sidor, S., and Wu, Y. Openai baselines. https:             Mannor, S., Simester, D., Sun, P., and Tsitsiklis, J. N. Bias
//github.com/openai/baselines, 2017.                                and variance approximation in value function estimates.
        Espeholt, L., Soyer, H., Munos, R., Simonyan, K., Mnih,     Management Science, 53(2):308-322, 2007.
V., Ward, T., Doron, Y., Firoiu, V., Harley, T., Dunning,             Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness,
I., et al. Impala: Scalable distributed deep-rl with impor-         J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidje-
tance weighted actor-learner architectures. arXiv preprint          land, A. K., Ostrovski, G., et al. Human-level control
arXiv:1802.01561, 2018.                                             through deep reinforcement learning. Nature, 518(7540):
                                                                    529-533, 2015.
        Fox, R., Pakman, A., and Tishby, N. Taming the noise in
reinforcement learning via soft updates. In Proceedings of              Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap,
the Thirty-Second Conference on Uncertainty in Artificial           T., Harley, T., Silver, D., and Kavukcuoglu, K. Asyn-
Intelligence, pp. 202-211. AUAI Press, 2016.                        chronous methods for deep reinforcement learning. In
                                                                    International Conference on Machine Learning, pp. 1928-
        Haarnoja, T., Zhou, A., Abbeel, P., and Levine, S. Soft     1937, 2016.
actor-critic: Off-policy maximum entropy deep reinforce-                Munos, R., Stepleton, T., Harutyunyan, A., and Bellemare,
ment learning with a stochastic actor. arXiv preprint               M. Safe and efficient off-policy reinforcement learning.
arXiv:1801.01290, 2018.                                             In Advances in Neural Information Processing Systems,
      He, F. S., Liu, Y., Schwing, A. G., and Peng, J. Learning     pp. 1054-1062, 2016.
to play in a day: Faster deep reinforcement learning by                   Nachum, O., Norouzi, M., Tucker, G., and Schuurmans, D.
optimality tightening. arXiv preprint arXiv:1611.01606,             Smoothed action value functions for learning gaussian
2016.                                                               policies. arXiv preprint arXiv:1803.02348, 2018.
     Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup,           O'Donoghue, B., Osband, I., Munos, R., and Mnih, V. The
D., and Meger, D. Deep Reinforcement Learning that                  uncertainty bellman equation and exploration.           arXiv
Matters. arXiv preprint arXiv:1709.06560, 2017.                     preprint arXiv:1709.05380, 2017.

     Horgan, D., Quan, J., Budden, D., Barth-Maron, G., Hessel,        Pendrith, M. D., Ryan, M. R., et al. Estimator variance in
M., van Hasselt, H., and Silver, D. Distributed prioritized         reinforcement learning: Theoretical problems and practi-
experience replay. International Conference on Learning             cal solutions. University of New South Wales, School of
Representations, 2018.                                              Computer Science and Engineering, 1997.

       Addressing Function Approximation Error in Actor-Critic Methods

        Petrik, M. and Scherrer, B. Biasing approximate dynamic    Van Seijen, H., Van Hasselt, H., Whiteson, S., and Wiering,
   programming with a lower discount factor. In Advances in         M. A theoretical and empirical analysis of expected sarsa.
   Neural Information Processing Systems, pp. 1265-1272,            In Adaptive Dynamic Programming and Reinforcement
   2009.                                                            Learning, 2009. ADPRL'09. IEEE Symposium on, pp.
        Popov, I., Heess, N., Lillicrap, T., Hafner, R., Barth-     177-184. IEEE, 2009.
   Maron, G., Vecerik, M., Lampe, T., Tassa, Y., Erez,                 Watkins, C. J. C. H. Learning from delayed rewards. PhD
   T., and Riedmiller, M. Data-efficient deep reinforce-            thesis, King's College, Cambridge, 1989.
   ment learning for dexterous manipulation. arXiv preprint
   arXiv:1704.03073, 2017.                                              Wu, Y., Mansimov, E., Grosse, R. B., Liao, S., and Ba,
                                                                    J. Scalable trust-region method for deep reinforcement
         Precup, D., Sutton, R. S., and Dasgupta, S. Off-policy     learning using kronecker-factored approximation. In Ad-
   temporal-difference learning with function approxima-            vances in Neural Information Processing Systems, pp.
   tion. In International Conference on Machine Learning,           5285-5294, 2017.
   pp. 417-424, 2001.

   Schaul, T., Quan, J., Antonoglou, I., and Silver, D. Priori-
   tized experience replay. In International Conference on
   Learning Representations, Puerto Rico, 2016.

  Schulman, J., Levine, S., Abbeel, P., Jordan, M., and Moritz,
   P. Trust region policy optimization. In International
   Conference on Machine Learning, pp. 1889-1897, 2015.

       Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and
   Klimov, O. Proximal policy optimization algorithms.
   arXiv preprint arXiv:1707.06347, 2017.

Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D., and
   Riedmiller, M. Deterministic policy gradient algorithms.
   In International Conference on Machine Learning, pp.
   387-395, 2014.

   Sutton, R. S. Learning to predict by the methods of temporal
   differences. Machine learning, 3(1):9-44, 1988.

      Sutton, R. S. and Barto, A. G. Reinforcement learning: An
   introduction, volume 1. MIT press Cambridge, 1998.

    Thrun, S. and Schwartz, A. Issues in using function approx-
   imation for reinforcement learning. In Proceedings of the
   1993 Connectionist Models Summer School Hillsdale, NJ.
   Lawrence Erlbaum, 1993.

         Todorov, E., Erez, T., and Tassa, Y. Mujoco: A physics
   engine for model-based control. In Intelligent Robots
   and Systems (IROS), 2012 IEEE/RSJ International Con-
   ference on, pp. 5026-5033. IEEE, 2012.

      Uhlenbeck, G. E. and Ornstein, L. S. On the theory of the
   brownian motion. Physical review, 36(5):823, 1930.

       Van Hasselt, H. Double q-learning. In Advances in Neural
   Information Processing Systems, pp. 2613-2621, 2010.

      Van Hasselt, H., Guez, A., and Silver, D. Deep reinforce-
   ment learning with double q-learning. In AAAI, pp. 2094-
   2100, 2016.