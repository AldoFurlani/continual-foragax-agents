from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import haiku as hk
import haiku.experimental as hkexp
import jax
import jax.numpy as jnp
import numpy as np

import utils.hk as hku

from algorithms.nn.rtus.rtus import RTLRTUs
import haiku.experimental.flax as hkflax

ModuleBuilder = Callable[[], Callable[[jax.Array | np.ndarray], jax.Array]]


def _get_core_layer_count(
    params: Dict[str, Any], new_name: str, legacy_name: str, default: int = 0
) -> int:
    """Read current config names while keeping older GRU-specific names valid."""
    return int(params.get(new_name, params.get(legacy_name, default)))


class NetworkBuilder:
    def __init__(self, input_shape: Tuple, params: Dict[str, Any], key: jax.Array):
        self._input_shape = tuple(input_shape)
        self._h_params = params
        self._rng, feat_rng = jax.random.split(key)

        self._feat_net, self._feat_params, self._feat_initializers = (
            buildFeatureNetwork(input_shape, params, feat_rng)
        )

        self._params = {
            "phi": self._feat_params,
        }

        self._initializers = {
            "phi": jax.tree.map(make_standalone_initializer, self._feat_initializers),
        }

        self._retrieved_params = False

        print(hkexp.tabulate(self._feat_net)(np.ones((1,) + self._input_shape)))

    def getParams(self):
        self._retrieved_params = True
        return self._params

    def getInitializers(self):
        """Get the initializers for each parameter group."""
        return self._initializers

    def getFeatureFunction(self):
        def _inner(params: Any, x: jax.Array | np.ndarray, **kwargs):
            return self._feat_net.apply(params["phi"], x, **kwargs)

        return _inner

    def getMultiplicativeActionRecurrentFeatureFunction(self):
        def _inner(
            params: Any,
            x: jax.Array | np.ndarray,
            a: jax.Array,
            reset: Optional[jax.Array | np.ndarray] = None,
            carry: Optional[jax.Array | np.ndarray] = None,
            is_target=False,
        ):
            return self._feat_net.apply(
                params["phi"], x, a, reset=reset, carry=carry, is_target=is_target
            )

        return _inner

    def getActionTraceMultiplicativeActionRecurrentFeatureFunction(self):
        def _inner(
            params: Any,
            x: jax.Array | np.ndarray,
            a: jax.Array,
            action_trace: jax.Array,
            reset: Optional[jax.Array | np.ndarray] = None,
            carry: Optional[jax.Array | np.ndarray] = None,
            is_target=False,
        ):
            return self._feat_net.apply(
                params["phi"],
                x,
                a,
                action_trace,
                reset=reset,
                carry=carry,
                is_target=is_target,
            )

        return _inner

    def addHead(
        self, module: ModuleBuilder, name: Optional[str] = None, grad: bool = True
    ):
        assert not self._retrieved_params, (
            "Attempted to add head after params have been retrieved"
        )
        _state = {}

        def _builder(x: jax.Array | np.ndarray):
            head = module()
            _state["name"] = getattr(head, "name", None)

            if not grad:
                x = jax.lax.stop_gradient(x)

            out = head(x)
            return out

        sample_in = jnp.zeros((1,) + self._input_shape)
        if "GRU" in self._h_params["type"] or "RTU" in self._h_params["type"]:
            sample_phi = self._feat_net.apply(self._feat_params, sample_in)[0]
        else:
            sample_phi = self._feat_net.apply(self._feat_params, sample_in).out
        self._sample_phi = sample_phi

        self._rng, rng = jax.random.split(self._rng)
        h_net = hk.without_apply_rng(hk.transform(_builder))
        h_params = h_net.init(rng, sample_phi)
        print(hkexp.tabulate(h_net)(sample_phi))

        name = name or _state.get("name")
        assert name is not None, "Could not detect name from module"
        self._params[name] = h_params

        # Default head initializer: TruncatedNormal for weights, zeros for biases
        def _get_initializer(path, _):
            # path is a tuple of keys, check if last key indicates weight or bias
            param_name = path[-1].key if hasattr(path[-1], "key") else str(path[-1])
            if param_name == "w":
                return hk.initializers.TruncatedNormal(
                    stddev=1.0 / np.sqrt(self._sample_phi.shape[-1])
                )
            elif param_name == "b":
                return hk.initializers.Constant(0)
            else:
                # Default for other parameters
                return hk.initializers.TruncatedNormal(
                    stddev=1.0 / np.sqrt(self._sample_phi.shape[-1])
                )

        self._initializers[name] = jax.tree_util.tree_map_with_path(
            _get_initializer, h_params
        )
        self._initializers[name] = jax.tree.map(
            make_standalone_initializer, self._initializers[name]
        )

        def _inner(params: Any, x: jax.Array):
            return h_net.apply(params[name], x)

        return h_net, h_params, _inner

    def reset(self, key: jax.Array):
        sample_in = jnp.zeros((1,) + self._input_shape)
        return {
            "phi": self._feat_net.init(key, sample_in),
        }


def reluLayers(
    layers: List[int],
    name: Optional[str] = None,
    layer_norm: bool = False,
    w_init: Optional[hk.initializers.Initializer] = None,
    b_init: Optional[hk.initializers.Initializer] = None,
):
    if w_init is None:
        w_init = hk.initializers.Orthogonal(np.sqrt(2))
    if b_init is None:
        b_init = hk.initializers.Constant(0)

    out = []
    for width in layers:
        out.append(hk.Linear(width, w_init=w_init, b_init=b_init, name=name))
        if layer_norm:
            out.append(hk.LayerNorm(axis=-1, create_scale=True, create_offset=True))
        out.append(jax.nn.relu)

    return out


def creluLayers(
    layers: List[int],
    name: Optional[str] = None,
    w_init: Optional[hk.initializers.Initializer] = None,
    b_init: Optional[hk.initializers.Initializer] = None,
):
    if w_init is None:
        w_init = hk.initializers.Orthogonal(np.sqrt(2))
    if b_init is None:
        b_init = hk.initializers.Constant(0)

    out = []
    for width in layers:
        out.append(hk.Linear(width, w_init=w_init, b_init=b_init, name=name))
        out.append(hku.crelu)

    return out


def buildFeatureNetwork(inputs: Tuple, params: Dict[str, Any], rng: Any):
    # Default initializer used across most networks
    w_init = hk.initializers.Orthogonal(np.sqrt(2))

    def _inner(x: jax.Array, *args, **kwargs):
        name = params["type"]
        hidden = params["hidden"]
        d_hidden = params.get("d_hidden", hidden)

        if name == "TwoLayerRelu":
            layers = reluLayers([hidden, hidden], name="phi")

        elif name == "OneLayerRelu":
            layers = reluLayers([hidden], name="phi")

        elif name == "TwoLayerCrelu":
            layers = creluLayers([hidden, hidden], name="phi")

        elif name == "OneLayerCrelu":
            layers = creluLayers([hidden], name="phi")

        elif name == "MinatarNet":
            w_init = hk.initializers.Orthogonal(np.sqrt(2))
            layers = [
                hk.Conv2D(16, 3, 2, w_init=w_init, name="phi"),
                jax.nn.relu,
                hk.Flatten(name="phi"),
            ]

        elif name == "ForagerNet":
            pre_core_layers = _get_core_layer_count(
                params, "pre_core_layers", "pre_gru_layers"
            )
            post_core_layers = _get_core_layer_count(
                params, "post_core_layers", "post_gru_layers"
            )
            net = ForagerNet(
                hidden=hidden,
                d_hidden=d_hidden,
                scalars=params["scalars"],
                layers=params.get("layers"),
                pre_core_layers=pre_core_layers,
                core_layers=params.get(
                    "core_layers",
                    1 if "layers" not in params and (pre_core_layers or post_core_layers) else 0,
                ),
                post_core_layers=post_core_layers,
                use_layernorm=params.get("use_layernorm", False),
                preactivation_layer_norm=params.get("preactivation_layer_norm", True),
                balanced=params.get("balanced", False),
                activation=params.get("activation", "relu"),
                name="phi",
                conv=params.get("conv", "Conv2D"),
                coord=params.get("coord", False),
            )
            return net(x, *args, **kwargs)

        elif name == "ForagerLayerNormNet":
            w_init = hk.initializers.Orthogonal(np.sqrt(2))
            layers = [
                hk.Conv2D(16, 3, 1, w_init=w_init, name="phi"),
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                jax.nn.relu,
                hk.Flatten(name="phi"),
            ]

        elif name == "Forager2Net":
            w_init = hk.initializers.Orthogonal(np.sqrt(2))
            layers = [
                hk.Conv2D(16, 3, 1, w_init=w_init, name="phi"),
                jax.nn.relu,
                hk.Flatten(name="phi"),
            ]
            layers += reluLayers([hidden, hidden], name="phi")

        elif name == "Forager2CreluNet":
            w_init = hk.initializers.Orthogonal(np.sqrt(2))
            layers = [
                hk.Conv2D(16, 3, 1, w_init=w_init, name="phi"),
                jax.nn.relu,
                hk.Flatten(name="phi"),
            ]
            layers += creluLayers([hidden, hidden], name="phi")

        elif name == "Forager2LayerNormNet":
            w_init = hk.initializers.Orthogonal(np.sqrt(2))
            layers = [
                hk.Conv2D(16, 3, 1, w_init=w_init, name="phi"),
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                jax.nn.relu,
                hk.Flatten(name="phi"),
            ]
            layers += reluLayers([hidden, hidden], name="phi", layer_norm=True)

        elif name == "ForagerGRUNetReLU":
            # It uses initializer different from above
            net = ForagerGRUNetReLU(
                hidden=hidden,
                d_hidden=d_hidden,
                scalars=params["scalars"],
                hint_size=params.get("hint_size", 0),
                hint_gru_only=params.get("hint_gru_only", False),
                balanced=params.get("balanced", False),
                pre_core_layers=_get_core_layer_count(
                    params, "pre_core_layers", "pre_gru_layers"
                ),
                post_core_layers=_get_core_layer_count(
                    params, "post_core_layers", "post_gru_layers"
                ),
                learn_initial_h=params.get("learn_initial_h", True),
                use_layernorm=params.get("use_layernorm", False),
                mlp=params.get("mlp", False),
                activation=params.get("activation", "relu"),
                name="ForagerGRUNetReLU",
            )
            return net(x, *args, **kwargs)

        elif name == "ForagerRTUNetReLU":
            # It uses initializer different from above
            net = ForagerRTUNetReLU(
                hidden=hidden,
                d_hidden=d_hidden,
                scalars=params["scalars"],
                hint_size=params.get("hint_size", 0),
                hint_gru_only=params.get("hint_gru_only", False),
                balanced=params.get("balanced", False),
                pre_core_layers=_get_core_layer_count(
                    params, "pre_core_layers", "pre_gru_layers"
                ),
                post_core_layers=_get_core_layer_count(
                    params, "post_core_layers", "post_gru_layers"
                ),
                learn_initial_h=params.get("learn_initial_h", True),
                use_layernorm=params.get("use_layernorm", False),
                mlp=params.get("mlp", False),
                activation=params.get("activation", "relu"),
                name="ForagerRTUNetReLU",
            )
            return net(x, *args, **kwargs)

        elif name == "ForagerMAGRUNetReLU":
            # It uses initializer different from above
            net = ForagerMAGRUNetReLU(
                hidden=hidden,
                actions=params["actions"],
                learn_initial_h=params.get("learn_initial_h", True),
                name="ForagerMAGRUNetReLU",
            )
            return net(x, *args, **kwargs)

        elif name == "ForagerAAGRUNetReLU":
            # It uses initializer different from above
            net = ForagerAAGRUNetReLU(
                hidden=hidden,
                actions=params["actions"],
                learn_initial_h=params.get("learn_initial_h", True),
                name="ForagerAAGRUNetReLU",
            )
            return net(x, *args, **kwargs)

        elif name == "ForagerATAAGRUNetReLU":
            # It uses initializer different from above
            net = ForagerATAAGRUNetReLU(
                hidden=hidden,
                actions=params["actions"],
                learn_initial_h=params.get("learn_initial_h", True),
                name="ForagerATAAGRUNetReLU",
            )
            return net(x, *args, **kwargs)

        elif name == "AtariNet":
            w_init = hk.initializers.Orthogonal(np.sqrt(2))
            layers = [
                lambda x: x.astype(np.float32),
                make_conv(32, (8, 8), (4, 4)),
                jax.nn.relu,
                make_conv(64, (4, 4), (2, 2)),
                jax.nn.relu,
                make_conv(64, (3, 3), (1, 1)),
                jax.nn.relu,
                hk.Flatten(),
                hk.Linear(512, w_init=w_init),
                jax.nn.relu,
            ]

        else:
            raise NotImplementedError()

        return hku.accumulatingSequence(layers)(x)

    network = hk.without_apply_rng(hk.transform(_inner))

    sample_input = jnp.zeros((1,) + tuple(inputs))
    net_params = network.init(rng, sample_input)

    # Feature network initializer: w_init for weights, zeros for biases
    def _get_feature_initializer(path, _):
        # path is a tuple of keys, check if last key indicates weight or bias
        param_name = path[-1].key if hasattr(path[-1], "key") else str(path[-1])
        if param_name == "w":
            return w_init
        elif param_name == "b":
            return hk.initializers.Constant(0)
        else:
            # Default for other parameters (e.g., LayerNorm scale/offset)
            return w_init

    inits = jax.tree_util.tree_map_with_path(_get_feature_initializer, net_params)

    return network, net_params, inits


def make_conv(size: int, shape: Tuple[int, int], stride: Tuple[int, int]):
    w_init = hk.initializers.Orthogonal(np.sqrt(2))
    b_init = hk.initializers.Constant(0)
    return hk.Conv2D(
        size,
        kernel_shape=shape,
        stride=stride,
        w_init=w_init,
        b_init=b_init,
        padding="VALID",
        name="conv",
    )


class RTU(hk.Module):
    def __init__(
        self,
        hidden: int,
        d_hidden: int,
        activation: str = "relu",
        name: str = "",
    ):
        super().__init__(name=name)
        self.hidden = hidden
        self.d_hidden = d_hidden
        self.activation = activation
        self.rtu = hkflax.lift(
            RTLRTUs(
                int(self.d_hidden),
                d_input=self.hidden,
                params_type="exp_exp",
                activation=activation,
                name="rtu_inner",
            ),
            name="lifted_rtu",
        )

    def initial_state(self, batch=1):
        hidden_init = (
            jnp.zeros((batch, self.d_hidden)),
            jnp.zeros((batch, self.d_hidden)),
        )
        memory_grad_init = (
            jnp.zeros((batch, self.d_hidden)),
            jnp.zeros((batch, self.d_hidden)),
            jnp.zeros((batch, self.d_hidden)),
            jnp.zeros((batch, self.d_hidden)),
            jnp.zeros((batch, self.hidden, self.d_hidden)),
            jnp.zeros((batch, self.hidden, self.d_hidden)),
            jnp.zeros((batch, self.hidden, self.d_hidden)),
            jnp.zeros((batch, self.hidden, self.d_hidden)),
        )
        return (hidden_init, memory_grad_init)

    def __call__(
        self,
        x: jnp.ndarray,
        reset: Optional[jnp.ndarray] = None,
        carry: Optional[Any] = None,
        is_target=False,
    ) -> tuple[jnp.ndarray, Any, Any]:
        """
        Args:
          x: Input tensor with shape [N, 1, ...]
          reset: Optional binary flag sequence with shape [N, 1] indicating when to reset the RTU state.
                 For example, at episode boundaries.
          carry: The initial hidden state for RNN.
          is_target: Target-network calls use the replayed/burned-in next-state carry directly
                     and intentionally do not overwrite it with the initial state on reset.

        Returns:
          outputs_sequence: Representation vectors sequence.
          states_sequence: The hidden states sequence.
        """

        N, T, *_ = x.shape

        assert T == 1, "RTU is designed to process one timestep at a time. Received input with T > 1."

        x = x.squeeze(1)  # Remove the time dimension since RTU processes one step at a time

        if reset is None:
            reset = jnp.zeros((N, T), dtype=bool)
        if carry is None:
            carry = self.initial_state(N)
        if carry[0][0].ndim != 2:
            carry = jax.tree.map(lambda c: c.squeeze(1).squeeze(1), carry)

        def broadcast_reset(mask, target):
            # Calculate how many extra dimensions the target has compared to the mask
            dims_to_add = target.ndim - mask.ndim
            # Append trailing 1s: e.g., (4, 1) -> (4, 1, 1)
            expanded_mask = mask.reshape(mask.shape + (1,) * dims_to_add)
            return expanded_mask

        # Online RTU calls start a new episode from the RTU initial state. Target calls
        # receive carryp, which is already the replayed or burn-in-derived next-state
        # carry, so resetting here would clobber the target boundary state.
        if not is_target:
            init_state = self.initial_state(N)
            carry = jax.tree.map(
                lambda init, curr: jnp.where(broadcast_reset(reset, curr), init, curr), init_state, carry
            )

        carry, output = self.rtu(carry, x)
        assert type(carry) is type(self.initial_state(1)), f"Expected carry type {type(self.initial_state(1))}, but got {type(carry)}"
        # Return both outputs and hidden states across the entire sequence.
        return output, carry, self.initial_state(1)

class GRU(hk.Module):
    def __init__(
        self,
        hidden: int,
        learn_initial_h=True,
        w_init=None,
        name: str = "",
    ):
        super().__init__(name=name)
        if w_init is None:
            w_init = hk.initializers.Orthogonal(np.sqrt(2))
        self.hidden = hidden
        self.gru = hk.GRU(
            self.hidden, name="gru_inner", w_h_init=w_init, w_i_init=w_init
        )
        self.learn_initial_h = learn_initial_h

    def initial_state(self, batch=1, length=1):
        if self.learn_initial_h:
            init_h = hk.get_parameter("initial_h", shape=(self.hidden,), init=jnp.zeros)
            init_h = jnp.repeat(init_h[None, :], batch, axis=0)
            init_h = jnp.repeat(init_h[:, None, :], length, axis=1)
        else:
            # This is all zeros
            init_h = jnp.repeat(
                self.gru.initial_state(batch_size=batch)[:, None, :], length, axis=1
            )
        return init_h

    def gru_step(self, prev_state, inputs):
        frame_feat, reset_flag, carry = inputs
        # Reset state if flag is True.
        prev_state = jax.lax.select(reset_flag, carry[None, :], prev_state)
        # GRU expects inputs with a batch dimension.
        output, next_state = self.gru(frame_feat[None, :], prev_state)
        # Remove the extra batch dimension and return both output and next_state.
        return next_state, (output[0], next_state[0])

    def process_sequence(self, features_seq, reset_seq, carry_seq):
        final_state, (outputs_seq, state_seq) = hk.scan(
            self.gru_step, carry_seq[:1, :], (features_seq, reset_seq, carry_seq)
        )
        return outputs_seq, state_seq

    def __call__(
        self,
        x: jnp.ndarray,
        reset: Optional[jnp.ndarray] = None,
        carry: Optional[jnp.ndarray] = None,
        is_target=False,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Args:
          x: Input tensor with shape [N, T, ...]
          reset: Optional binary flag sequence with shape [N, T] indicating when to reset the GRU state.
                 For example, at episode boundaries.
          carry: The initial hidden state for RNN.

        Returns:
          outputs_sequence: Representation vectors sequence.
          states_sequence: The hidden states sequence.
        """

        N, T, *_ = x.shape

        if reset is None:
            reset = jnp.zeros((N, T), dtype=bool)

        if carry is None:
            carry = self.initial_state(N, T)
        elif len(carry.shape) < 3:
            carry = carry[:, None, :]

        # Replace entries in carry where reset is true with the initial state
        if self.learn_initial_h and not is_target:
            init_state = self.initial_state(N, T)
            carry = jnp.where(reset[..., None], init_state, carry)

        # Vectorize the per-sequence unroll over the batch dimension.
        # x has shape [N, T, ...] and reset has shape [N, T].
        outputs_sequence, states_sequence = jax.vmap(self.process_sequence)(
            x, reset, carry
        )

        # Return both the GRU outputs and hidden states across the entire sequence.
        return outputs_sequence, states_sequence, self.initial_state(1, 1)[:, 0, ...]


class MAGRU(hk.Module):
    def __init__(self, hidden: int, actions: int, learn_initial_h=True, name: str = ""):
        super().__init__(name=name)
        self.hidden_size = hidden
        self.number_of_actions = actions
        self.w_init = hk.initializers.Orthogonal(np.sqrt(2))
        self.b_init = jnp.zeros
        self.learn_initial_h = learn_initial_h

    def initial_state(self, batch=1, length=1):
        if self.learn_initial_h:
            init_h = hk.get_parameter(
                "initial_h", shape=(self.hidden_size,), init=jnp.zeros
            )
            init_h = jnp.repeat(init_h[None, :], batch, axis=0)
            init_h = jnp.repeat(init_h[:, None, :], length, axis=1)
        else:
            # This is all zeros
            init_h = jnp.broadcast_to(
                jnp.zeros([self.hidden_size]), (batch, length, self.hidden_size)
            )
        return init_h

    def gru_call(self, inputs, action, state):
        # modified from https://github.com/google-deepmind/dm-haiku/blob/main/haiku/_src/recurrent.py#L521#L588
        self.input_size = inputs.shape[-1]
        w_i = hk.get_parameter(
            "w_i",
            [self.number_of_actions, self.input_size, 3 * self.hidden_size],
            init=self.w_init,
        )
        w_h = hk.get_parameter(
            "w_h",
            [self.number_of_actions, self.hidden_size, 3 * self.hidden_size],
            init=self.w_init,
        )
        b = hk.get_parameter(
            "b", [self.number_of_actions, 3 * self.hidden_size], init=self.b_init
        )
        w_i = w_i[action]
        w_h = w_h[action]
        b = b[action]
        w_h_z, w_h_a = jnp.split(
            w_h, indices_or_sections=[2 * self.hidden_size], axis=1
        )
        b_z, b_a = jnp.split(b, indices_or_sections=[2 * self.hidden_size], axis=0)

        gates_x = jnp.matmul(inputs, w_i)
        zr_x, a_x = jnp.split(
            gates_x, indices_or_sections=[2 * self.hidden_size], axis=-1
        )
        zr_h = jnp.matmul(state, w_h_z)
        zr = zr_x + zr_h + jnp.broadcast_to(b_z, zr_h.shape)
        z, r = jnp.split(jax.nn.sigmoid(zr), indices_or_sections=2, axis=-1)

        a_h = jnp.matmul(r * state, w_h_a)
        a = jnp.tanh(a_x + a_h + jnp.broadcast_to(b_a, a_h.shape))

        next_state = (1 - z) * state + z * a
        return next_state, next_state

    def gru_step(self, prev_state, inputs):
        frame_feat, action, reset_flag, carry = inputs
        # Reset state if flag is True.
        prev_state = jax.lax.select(reset_flag, carry[None, :], prev_state)
        # GRU expects inputs with a batch dimension.
        output, next_state = self.gru_call(frame_feat[None, :], action, prev_state)
        # Remove the extra batch dimension and return both output and next_state.
        return next_state, (output[0], next_state[0])

    def process_sequence(self, features_seq, action_seq, reset_seq, carry_seq):
        final_state, (outputs_seq, state_seq) = hk.scan(
            self.gru_step,
            carry_seq[:1, :],
            (features_seq, action_seq, reset_seq, carry_seq),
        )
        return outputs_seq, state_seq

    def __call__(
        self,
        x: jnp.ndarray,
        a: jnp.ndarray,
        reset: Optional[jnp.ndarray] = None,
        carry: Optional[jnp.ndarray] = None,
        is_target=False,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Args:
          x: Input tensor with shape [N, T, ...]
          a: Action tensor with shape [N, T]
          reset: Optional binary flag sequence with shape [N, T] indicating when to reset the GRU state.
                 For example, at episode boundaries.
          carry: The initial hidden state for RNN.

        Returns:
          outputs_sequence: Representation vectors sequence.
          states_sequence: The hidden states sequence.
        """

        N, T, *_ = x.shape

        if reset is None:
            reset = jnp.zeros((N, T), dtype=bool)

        if carry is None:
            carry = self.initial_state(N, T)
        elif len(carry.shape) < 3:
            carry = carry[:, None, :]

        # Replace entries in carry where reset is true with the initial state
        if self.learn_initial_h and not is_target:
            init_state = self.initial_state(N, T)
            carry = jnp.where(reset[..., None], init_state, carry)

        # Vectorize the per-sequence unroll over the batch dimension.
        # x has shape [N, T, ...], a has shape [N, T], and reset has shape [N, T].
        outputs_sequence, states_sequence = jax.vmap(self.process_sequence)(
            x, a, reset, carry
        )

        # Return both the GRU outputs and hidden states across the entire sequence.
        return outputs_sequence, states_sequence, self.initial_state(1, 1)[:, 0, ...]


class AAGRU(hk.Module):
    def __init__(self, hidden: int, actions: int, learn_initial_h=True, name: str = ""):
        super().__init__(name=name)
        self.hidden_size = hidden
        self.number_of_actions = actions
        self.w_init = hk.initializers.Orthogonal(np.sqrt(2))
        self.b_init = jnp.zeros
        self.learn_initial_h = learn_initial_h

    def initial_state(self, batch=1, length=1):
        if self.learn_initial_h:
            init_h = hk.get_parameter(
                "initial_h", shape=(self.hidden_size,), init=jnp.zeros
            )
            init_h = jnp.repeat(init_h[None, :], batch, axis=0)
            init_h = jnp.repeat(init_h[:, None, :], length, axis=1)
        else:
            # This is all zeros
            init_h = jnp.broadcast_to(
                jnp.zeros([self.hidden_size]), (batch, length, self.hidden_size)
            )
        return init_h

    def gru_call(self, inputs, action, state):
        # modified from https://github.com/google-deepmind/dm-haiku/blob/main/haiku/_src/recurrent.py#L521#L588
        self.input_size = inputs.shape[-1]
        w_i = hk.get_parameter(
            "w_i", [self.input_size, 3 * self.hidden_size], init=self.w_init
        )
        w_h = hk.get_parameter(
            "w_h", [self.hidden_size, 3 * self.hidden_size], init=self.w_init
        )
        w_a = hk.get_parameter(
            "w_a", [self.number_of_actions, 3 * self.hidden_size], init=self.w_init
        )
        b = hk.get_parameter("b", [3 * self.hidden_size], init=self.b_init)

        w_h_z, w_h_a = jnp.split(
            w_h, indices_or_sections=[2 * self.hidden_size], axis=1
        )
        b_z, b_a = jnp.split(b, indices_or_sections=[2 * self.hidden_size], axis=0)

        gates_x = jnp.matmul(inputs, w_i)
        gates_a = jnp.matmul(action, w_a)

        zr_x, a_x = jnp.split(
            gates_x, indices_or_sections=[2 * self.hidden_size], axis=-1
        )
        zr_a, a_a = jnp.split(
            gates_a, indices_or_sections=[2 * self.hidden_size], axis=-1
        )

        zr_h = jnp.matmul(state, w_h_z)

        zr = zr_x + zr_h + zr_a + jnp.broadcast_to(b_z, zr_h.shape)
        z, r = jnp.split(jax.nn.sigmoid(zr), indices_or_sections=2, axis=-1)

        a_h = jnp.matmul(r * state, w_h_a)
        a = jnp.tanh(a_x + a_h + a_a + jnp.broadcast_to(b_a, a_h.shape))

        next_state = (1 - z) * state + z * a
        return next_state, next_state

    def gru_step(self, prev_state, inputs):
        frame_feat, action, reset_flag, carry = inputs
        # Reset state if flag is True.
        prev_state = jax.lax.select(reset_flag, carry[None, :], prev_state)
        # GRU expects inputs with a batch dimension.
        output, next_state = self.gru_call(frame_feat[None, :], action, prev_state)
        # Remove the extra batch dimension and return both output and next_state.
        return next_state, (output[0], next_state[0])

    def process_sequence(self, features_seq, action_seq, reset_seq, carry_seq):
        final_state, (outputs_seq, state_seq) = hk.scan(
            self.gru_step,
            carry_seq[:1, :],
            (features_seq, action_seq, reset_seq, carry_seq),
        )
        return outputs_seq, state_seq

    def __call__(
        self,
        x: jnp.ndarray,
        a: jnp.ndarray,
        reset: Optional[jnp.ndarray] = None,
        carry: Optional[jnp.ndarray] = None,
        is_target=False,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Args:
          x: Input tensor with shape [N, T, ...]
          a: Action tensor with shape [N, T]
          reset: Optional binary flag sequence with shape [N, T] indicating when to reset the GRU state.
                 For example, at episode boundaries.
          carry: The initial hidden state for RNN.

        Returns:
          outputs_sequence: Representation vectors sequence.
          states_sequence: The hidden states sequence.
        """

        N, T, *_ = x.shape

        if reset is None:
            reset = jnp.zeros((N, T), dtype=bool)

        if carry is None:
            carry = self.initial_state(N, T)
        elif len(carry.shape) < 3:
            carry = carry[:, None, :]

        # Replace entries in carry where reset is true with the initial state
        if self.learn_initial_h and not is_target:
            init_state = self.initial_state(N, T)
            carry = jnp.where(reset[..., None], init_state, carry)

        # Vectorize the per-sequence unroll over the batch dimension.
        # x has shape [N, T, ...], a has shape [N, T, num_action], and reset has shape [N, T].
        outputs_sequence, states_sequence = jax.vmap(self.process_sequence)(
            x, a, reset, carry
        )

        # Return both the GRU outputs and hidden states across the entire sequence.
        return outputs_sequence, states_sequence, self.initial_state(1, 1)[:, 0, ...]


class ATAAGRU(hk.Module):
    def __init__(self, hidden: int, actions: int, learn_initial_h=True, name: str = ""):
        super().__init__(name=name)
        self.hidden_size = hidden
        self.number_of_actions = actions
        self.w_init = hk.initializers.Orthogonal(np.sqrt(2))
        self.b_init = jnp.zeros
        self.learn_initial_h = learn_initial_h

    def initial_state(self, batch=1, length=1):
        if self.learn_initial_h:
            init_h = hk.get_parameter(
                "initial_h", shape=(self.hidden_size,), init=jnp.zeros
            )
            init_h = jnp.repeat(init_h[None, :], batch, axis=0)
            init_h = jnp.repeat(init_h[:, None, :], length, axis=1)
        else:
            # This is all zeros
            init_h = jnp.broadcast_to(
                jnp.zeros([self.hidden_size]), (batch, length, self.hidden_size)
            )
        return init_h

    def gru_call(self, inputs, action, action_trace, state):
        # modified from https://github.com/google-deepmind/dm-haiku/blob/main/haiku/_src/recurrent.py#L521#L588
        self.input_size = inputs.shape[-1]
        w_i = hk.get_parameter(
            "w_i", [self.input_size, 3 * self.hidden_size], init=self.w_init
        )
        w_h = hk.get_parameter(
            "w_h", [self.hidden_size, 3 * self.hidden_size], init=self.w_init
        )
        w_a = hk.get_parameter(
            "w_a", [self.number_of_actions, 3 * self.hidden_size], init=self.w_init
        )
        w_at = hk.get_parameter(
            "w_at", [self.number_of_actions, 3 * self.hidden_size], init=self.w_init
        )
        b = hk.get_parameter("b", [3 * self.hidden_size], init=self.b_init)

        w_h_z, w_h_a = jnp.split(
            w_h, indices_or_sections=[2 * self.hidden_size], axis=1
        )
        b_z, b_a = jnp.split(b, indices_or_sections=[2 * self.hidden_size], axis=0)

        gates_x = jnp.matmul(inputs, w_i)
        gates_a = jnp.matmul(action, w_a)
        gates_at = jnp.matmul(action, w_at)

        zr_x, a_x = jnp.split(
            gates_x, indices_or_sections=[2 * self.hidden_size], axis=-1
        )
        zr_a, a_a = jnp.split(
            gates_a, indices_or_sections=[2 * self.hidden_size], axis=-1
        )
        zr_at, a_at = jnp.split(
            gates_at, indices_or_sections=[2 * self.hidden_size], axis=-1
        )

        zr_h = jnp.matmul(state, w_h_z)

        zr = zr_x + zr_h + zr_a + zr_at + jnp.broadcast_to(b_z, zr_h.shape)
        z, r = jnp.split(jax.nn.sigmoid(zr), indices_or_sections=2, axis=-1)

        a_h = jnp.matmul(r * state, w_h_a)
        a = jnp.tanh(a_x + a_h + a_a + a_at + jnp.broadcast_to(b_a, a_h.shape))

        next_state = (1 - z) * state + z * a
        return next_state, next_state

    def gru_step(self, prev_state, inputs):
        frame_feat, action, action_trace, reset_flag, carry = inputs
        # Reset state if flag is True.
        prev_state = jax.lax.select(reset_flag, carry[None, :], prev_state)
        # GRU expects inputs with a batch dimension.
        output, next_state = self.gru_call(
            frame_feat[None, :], action, action_trace, prev_state
        )
        # Remove the extra batch dimension and return both output and next_state.
        return next_state, (output[0], next_state[0])

    def process_sequence(
        self, features_seq, action_seq, action_trace_seq, reset_seq, carry_seq
    ):
        final_state, (outputs_seq, state_seq) = hk.scan(
            self.gru_step,
            carry_seq[:1, :],
            (features_seq, action_seq, action_trace_seq, reset_seq, carry_seq),
        )
        return outputs_seq, state_seq

    def __call__(
        self,
        x: jnp.ndarray,
        a: jnp.ndarray,
        action_trace: jnp.ndarray,
        reset: Optional[jnp.ndarray] = None,
        carry: Optional[jnp.ndarray] = None,
        is_target=False,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Args:
          x: Input tensor with shape [N, T, ...]
          a: Action tensor with shape [N, T]
          reset: Optional binary flag sequence with shape [N, T] indicating when to reset the GRU state.
                 For example, at episode boundaries.
          carry: The initial hidden state for RNN.

        Returns:
          outputs_sequence: Representation vectors sequence.
          states_sequence: The hidden states sequence.
        """

        N, T, *_ = x.shape

        if reset is None:
            reset = jnp.zeros((N, T), dtype=bool)

        if carry is None:
            carry = self.initial_state(N, T)
        elif len(carry.shape) < 3:
            carry = carry[:, None, :]

        # Replace entries in carry where reset is true with the initial state
        if self.learn_initial_h and not is_target:
            init_state = self.initial_state(N, T)
            carry = jnp.where(reset[..., None], init_state, carry)

        # Vectorize the per-sequence unroll over the batch dimension.
        # x has shape [N, T, ...], a has shape [N, T, num_action], and reset has shape [N, T].
        outputs_sequence, states_sequence = jax.vmap(self.process_sequence)(
            x, a, action_trace, reset, carry
        )

        # Return both the GRU outputs and hidden states across the entire sequence.
        return outputs_sequence, states_sequence, self.initial_state(1, 1)[:, 0, ...]


class ForagerGRUNetReLU(hk.Module):
    def __init__(
        self,
        hidden: int,
        d_hidden: int,
        scalars: int = 0,
        hint_size: int = 0,
        hint_gru_only: bool = False,
        balanced: bool = False,
        pre_core_layers: Optional[int] = None,
        post_core_layers: Optional[int] = None,
        pre_gru_layers: int = 0,
        post_gru_layers: int = 0,
        learn_initial_h=True,
        use_layernorm=False,
        mlp: bool = False,
        activation: str = "relu",
        name: str = "",
    ):
        super().__init__(name=name)
        self.hidden = hidden
        self.d_hidden = d_hidden
        self.scalars = scalars
        self.hint_size = hint_size
        self.hint_gru_only = hint_gru_only
        self.balanced = balanced
        self.other_scalars = scalars - hint_size
        self.use_layernorm = use_layernorm
        self.pre_core_layers = (
            pre_core_layers if pre_core_layers is not None else pre_gru_layers
        )
        self.post_core_layers = (
            post_core_layers if post_core_layers is not None else post_gru_layers
        )
        self.pre_gru_layers = self.pre_core_layers
        self.post_gru_layers = self.post_core_layers
        self.mlp = mlp
        self.activation_fn = hku.crelu if activation == "crelu" else jax.nn.relu
        w_init = hk.initializers.Orthogonal(np.sqrt(2))

        if not mlp:
            embedding = [hk.Conv2D(16, 3, 1, w_init=w_init, name="phi")]
            if use_layernorm:
                embedding.append(
                    hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
                )
            embedding.append(self.activation_fn)
            self.embedding = hk.Sequential(embedding)

        self.flatten = hk.Flatten(preserve_dims=2, name="flatten")

        # Balanced mode: project vision and scalars to equal-sized embeddings
        if self.balanced:
            vision_proj = [hk.Linear(self.hidden, w_init=w_init, name="vision_proj")]
            if use_layernorm:
                vision_proj.append(
                    hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
                )
            vision_proj.append(self.activation_fn)
            self.vision_proj = hk.Sequential(vision_proj)

            if self.scalars > 0:
                scalars_proj = [hk.Linear(self.hidden, w_init=w_init, name="scalars_proj")]
                if use_layernorm:
                    scalars_proj.append(
                        hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
                    )
                scalars_proj.append(self.activation_fn)
                self.scalars_proj = hk.Sequential(scalars_proj)

        if self.pre_core_layers > 0:
            layers = []
            for _ in range(self.pre_core_layers):
                layers.append(hk.Linear(self.hidden, w_init=w_init))
                if use_layernorm:
                    layers.append(
                        hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
                    )
                layers.append(self.activation_fn)
            self.pre_core_mlp = hk.Sequential(layers)

        self.gru = GRU(self.d_hidden, learn_initial_h=learn_initial_h, name="gru")

        if self.post_core_layers > 0:
            layers = []
            for _ in range(self.post_core_layers):
                layers.append(hk.Linear(self.hidden, w_init=w_init))
                if use_layernorm:
                    layers.append(
                        hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
                    )
                layers.append(self.activation_fn)
            self.post_core_mlp = hk.Sequential(layers)

        self.phi = hk.Flatten(preserve_dims=2, name="phi")

    def __call__(
        self,
        x: jnp.ndarray,
        scalars: Optional[jnp.ndarray] = None,
        reset: Optional[jnp.ndarray] = None,
        carry: Optional[jnp.ndarray] = None,
        is_target=False,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Args:
          x: Input tensor with shape [N, T, ...]
          scalars: Optional scalar features with shape [N, T, S]
          reset: Optional binary flag sequence with shape [N, T] indicating when to reset the GRU state.
          carry: The initial hidden state for RNN.
          hint_gru_only: When True, only the hint portion of scalars is fed through
                         the GRU while vision remains feedforward.

        Returns:
          outputs_sequence: Representation vectors sequence.
          states_sequence: The hidden states sequence.
        """
        # Add temporal dimension if given a single slice
        if len(x.shape) < 5:
            x = x[:, None]

        N, T, *feat = x.shape

        if not self.mlp:
            x = jnp.reshape(x, (N * T, *feat))
            x = self.embedding(x)
            _, *feat = x.shape
            x = jnp.reshape(x, (N, T, *feat))

        h = self.flatten(x)

        # Balanced mode: project vision and scalars to equal-sized embeddings
        if self.balanced:
            h = self.vision_proj(h)

        if self.scalars > 0:
            if scalars is None:
                scalars = jnp.zeros((N, T, self.scalars))
            elif len(scalars.shape) < 3:
                scalars = jnp.broadcast_to(scalars, (N, T, self.scalars))
            if self.balanced:
                scalars = self.scalars_proj(scalars)
            scalars = cast(jnp.ndarray, scalars)
            assert scalars is not None

        if self.hint_gru_only and self.hint_size > 0:
            # GRU on hint only; vision + other scalars stay feedforward
            if self.scalars > 0:
                assert scalars is not None
                other = scalars[..., : self.other_scalars]
                hint = scalars[..., self.other_scalars :]
            else:
                other = None
                hint = jnp.zeros((N, T, self.hint_size))

            gru_in = hint
            if self.pre_core_layers > 0:
                gru_in = self.pre_core_mlp(gru_in)

            gru_out, states_sequence, initial_carry = self.gru(
                gru_in, reset, carry, is_target=is_target
            )
            gru_out = self.activation_fn(gru_out)

            # Concat: vision + gru_out + skip(hint) + other_scalars
            parts = [h, gru_out, hint]
            if other is not None and self.other_scalars > 0:
                parts.append(other)
            outputs_sequence = jnp.concatenate(parts, axis=-1)
        else:
            # Standard: concat all then GRU on everything
            if self.scalars > 0:
                assert scalars is not None
                h = jnp.concatenate([h, scalars], axis=-1)

            if self.pre_core_layers > 0:
                h = self.pre_core_mlp(h)

            outputs_sequence, states_sequence, initial_carry = self.gru(
                h, reset, carry, is_target=is_target
            )
            outputs_sequence = self.activation_fn(outputs_sequence)
            outputs_sequence = jnp.concatenate([outputs_sequence, h], axis=-1)

        if self.post_core_layers > 0:
            outputs_sequence = self.post_core_mlp(outputs_sequence)

        outputs_sequence = self.phi(outputs_sequence)

        # Return both the GRU outputs and hidden states across the entire sequence along with initial hidden state
        return outputs_sequence, states_sequence, initial_carry

class ForagerRTUNetReLU(hk.Module):
    def __init__(
        self,
        hidden: int,
        d_hidden: int,
        scalars: int = 0,
        hint_size: int = 0,
        hint_gru_only: bool = False,
        balanced: bool = False,
        pre_core_layers: Optional[int] = None,
        post_core_layers: Optional[int] = None,
        pre_gru_layers: int = 0,
        post_gru_layers: int = 0,
        learn_initial_h=True,
        use_layernorm=False,
        mlp: bool = False,
        activation: str = "relu",
        name: str = "",
    ):
        super().__init__(name=name)
        self.hidden = hidden
        self.d_hidden = d_hidden
        self.scalars = scalars
        self.hint_size = hint_size
        self.hint_gru_only = hint_gru_only
        self.balanced = balanced
        self.other_scalars = scalars - hint_size
        self.use_layernorm = use_layernorm
        self.pre_core_layers = (
            pre_core_layers if pre_core_layers is not None else pre_gru_layers
        )
        self.post_core_layers = (
            post_core_layers if post_core_layers is not None else post_gru_layers
        )
        self.pre_gru_layers = self.pre_core_layers
        self.post_gru_layers = self.post_core_layers
        self.mlp = mlp
        self.activation = activation
        self.activation_fn = hku.crelu if activation == "crelu" else jax.nn.relu
        w_init = hk.initializers.Orthogonal(np.sqrt(2))

        if not mlp:
            embedding = [hk.Conv2D(16, 3, 1, w_init=w_init, name="phi")]
            if use_layernorm:
                embedding.append(
                    hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
                )
            embedding.append(self.activation_fn)
            self.embedding = hk.Sequential(embedding)

        self.flatten = hk.Flatten(preserve_dims=2, name="flatten")

        # Balanced mode: project vision and scalars to equal-sized embeddings
        if self.balanced:
            vision_proj = [hk.Linear(self.hidden, w_init=w_init, name="vision_proj")]
            if use_layernorm:
                vision_proj.append(
                    hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
                )
            vision_proj.append(self.activation_fn)
            self.vision_proj = hk.Sequential(vision_proj)

            if self.scalars > 0:
                scalars_proj = [hk.Linear(self.hidden, w_init=w_init, name="scalars_proj")]
                if use_layernorm:
                    scalars_proj.append(
                        hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
                    )
                scalars_proj.append(self.activation_fn)
                self.scalars_proj = hk.Sequential(scalars_proj)

        if self.pre_core_layers > 0:
            layers = []
            for _ in range(self.pre_core_layers):
                layers.append(hk.Linear(self.hidden, w_init=w_init))
                if use_layernorm:
                    layers.append(
                        hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
                    )
                layers.append(self.activation_fn)
            self.pre_core_mlp = hk.Sequential(layers)

        if self.post_core_layers > 0:
            layers = []
            for _ in range(self.post_core_layers):
                layers.append(hk.Linear(self.hidden, w_init=w_init))
                if use_layernorm:
                    layers.append(
                        hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
                    )
                layers.append(self.activation_fn)
            self.post_core_mlp = hk.Sequential(layers)

        self.phi = hk.Flatten(preserve_dims=2, name="phi")

    def __call__(
        self,
        x: jnp.ndarray,
        scalars: Optional[jnp.ndarray] = None,
        reset: Optional[jnp.ndarray] = None,
        carry: Optional[jnp.ndarray] = None,
        is_target=False,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Args:
          x: Input tensor with shape [N, T, ...]
          scalars: Optional scalar features with shape [N, T, S]
          reset: Optional binary flag sequence with shape [N, T] indicating when to reset the GRU state.
          carry: The initial hidden state for RNN.
          hint_gru_only: When True, only the hint portion of scalars is fed through
                         the GRU while vision remains feedforward.

        Returns:
          outputs_sequence: Representation vectors sequence.
          states_sequence: The hidden states sequence.
        """
        # Add temporal dimension if given a single slice
        if len(x.shape) < 5:
            x = x[:, None]

        N, T, *feat = x.shape

        assert T == 1, "RTU version only supports sequence length of 1 for now"

        if not self.mlp:
            x = jnp.reshape(x, (N * T, *feat))
            x = self.embedding(x)
            _, *feat = x.shape
            x = jnp.reshape(x, (N, T, *feat))

        h = self.flatten(x)

        # Balanced mode: project vision and scalars to equal-sized embeddings
        if self.balanced:
            h = self.vision_proj(h)

        if self.scalars > 0:
            if scalars is None:
                scalars = jnp.zeros((N, T, self.scalars))
            elif len(scalars.shape) < 3:
                scalars = jnp.broadcast_to(scalars, (N, T, self.scalars))
            if self.balanced:
                scalars = self.scalars_proj(scalars)
            scalars = cast(jnp.ndarray, scalars)
            assert scalars is not None

        if self.hint_gru_only and self.hint_size > 0:
            # GRU on hint only; vision + other scalars stay feedforward
            if self.scalars > 0:
                assert scalars is not None
                other = scalars[..., : self.other_scalars]
                hint = scalars[..., self.other_scalars :]
            else:
                other = None
                hint = jnp.zeros((N, T, self.hint_size))

            gru_in = hint
            if self.pre_core_layers > 0:
                gru_in = self.pre_core_mlp(gru_in)

            rtu = RTU(
                hidden=gru_in.shape[-1],
                d_hidden=self.d_hidden,
                activation=self.activation,
                name="rtu",
            )
            gru_out, states_sequence, initial_carry = rtu(
                gru_in, reset, carry, is_target=is_target
            )

            # Concat: vision + gru_out + skip(hint) + other_scalars
            parts = [h, gru_out[:, None], hint]
            if other is not None and self.other_scalars > 0:
                parts.append(other)
            outputs_sequence = jnp.concatenate(parts, axis=-1)
        else:
            if self.pre_core_layers > 0:
                h = self.pre_core_mlp(h)

            # Standard: concat scalars immediately before the recurrent core.
            if self.scalars > 0:
                assert scalars is not None
                core_in = jnp.concatenate([h, scalars], axis=-1)
            else:
                core_in = h

            rtu = RTU(
                hidden=core_in.shape[-1],
                d_hidden=self.d_hidden,
                activation=self.activation,
                name="rtu",
            )
            outputs_sequence, states_sequence, initial_carry = rtu(
                core_in, reset, carry, is_target=is_target
            )
            outputs_sequence = jnp.concatenate(
                [outputs_sequence[:, None], core_in], axis=-1
            )

        if self.post_core_layers > 0:
            outputs_sequence = self.post_core_mlp(outputs_sequence)

        outputs_sequence = self.phi(outputs_sequence)

        # Return both the GRU outputs and hidden states across the entire sequence along with initial hidden state
        return outputs_sequence, states_sequence, initial_carry

class ForagerAAGRUNetReLU(hk.Module):
    def __init__(self, hidden: int, actions: int, learn_initial_h=True, name: str = ""):
        super().__init__(name=name)
        self.hidden = hidden
        self.number_of_actions = actions
        w_init = hk.initializers.Orthogonal(np.sqrt(2))

        self.conv = hk.Conv2D(16, 3, 1, w_init=w_init, name="phi")

        self.flatten = hk.Flatten(preserve_dims=2, name="flatten")

        self.skip_connection = hk.Linear(
            self.hidden, w_init=w_init, name="skip_connection"
        )

        self.aagru = AAGRU(
            self.hidden,
            self.number_of_actions,
            learn_initial_h=learn_initial_h,
            name="aagru",
        )

        self.phi = hk.Flatten(preserve_dims=2, name="phi")

    def __call__(
        self,
        x: jnp.ndarray,
        a: Optional[jnp.ndarray] = None,
        reset: Optional[jnp.ndarray] = None,
        carry: Optional[jnp.ndarray] = None,
        is_target=False,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Args:
          x: Input tensor with shape [N, T, ...]
          reset: Optional binary flag sequence with shape [N, T] indicating when to reset the GRU state.
                 For example, at episode boundaries.
          carry: The initial hidden state for RNN.

        Returns:
          outputs_sequence: Representation vectors sequence.
          states_sequence: The hidden states sequence.
        """
        # Add temporal dimension if given a single slice
        if len(x.shape) < 5:
            x = x[:, None]

        N, T, *feat = x.shape

        # Use No-Op action 0 to populate a if None
        if a is None:
            a = jnp.full((N, T, self.number_of_actions), jnp.float32(0))
        if len(a.shape) < 3:
            a = jnp.broadcast_to(a, (N, T, self.number_of_actions))

        x = jnp.reshape(x, (N * T, *feat))

        h = self.conv(x)
        h = jax.nn.relu(h)

        _, *feat = h.shape

        h = jnp.reshape(h, (N, T, *feat))

        h = self.flatten(h)

        outputs_sequence, states_sequence, initial_carry = self.aagru(
            h, a, reset, carry, is_target=is_target
        )

        outputs_sequence = jax.nn.relu(outputs_sequence)

        outputs_sequence = outputs_sequence + self.skip_connection(h)

        outputs_sequence = self.phi(outputs_sequence)

        # Return both the GRU outputs and hidden states across the entire sequence along with initial hidden state
        return outputs_sequence, states_sequence, initial_carry


class ForagerATAAGRUNetReLU(hk.Module):
    def __init__(self, hidden: int, actions: int, learn_initial_h=True, name: str = ""):
        super().__init__(name=name)
        self.hidden = hidden
        self.number_of_actions = actions
        w_init = hk.initializers.Orthogonal(np.sqrt(2))

        self.conv = hk.Conv2D(16, 3, 1, w_init=w_init, name="phi")

        self.flatten = hk.Flatten(preserve_dims=2, name="flatten")

        self.skip_connection = hk.Linear(
            self.hidden, w_init=w_init, name="skip_connection"
        )

        self.ataagru = ATAAGRU(
            self.hidden,
            self.number_of_actions,
            learn_initial_h=learn_initial_h,
            name="aagru",
        )

        self.phi = hk.Flatten(preserve_dims=2, name="phi")

    def __call__(
        self,
        x: jnp.ndarray,
        a: Optional[jnp.ndarray] = None,
        action_trace: Optional[jnp.ndarray] = None,
        reset: Optional[jnp.ndarray] = None,
        carry: Optional[jnp.ndarray] = None,
        is_target=False,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Args:
          x: Input tensor with shape [N, T, ...]
          reset: Optional binary flag sequence with shape [N, T] indicating when to reset the GRU state.
                 For example, at episode boundaries.
          carry: The initial hidden state for RNN.

        Returns:
          outputs_sequence: Representation vectors sequence.
          states_sequence: The hidden states sequence.
        """
        # Add temporal dimension if given a single slice
        if len(x.shape) < 5:
            x = x[:, None]

        N, T, *feat = x.shape

        # Use No-Op action 0 to populate a if None
        if a is None:
            a = jnp.zeros((N, T, self.number_of_actions))
        if len(a.shape) < 3:
            a = jnp.broadcast_to(a, (N, T, self.number_of_actions))
        if action_trace is None:
            action_trace = jnp.full((N, T, self.number_of_actions), jnp.float32(0))
        if len(action_trace.shape) < 3:
            action_trace = jnp.broadcast_to(a, (N, T, self.number_of_actions))

        x = jnp.reshape(x, (N * T, *feat))

        h = self.conv(x)
        h = jax.nn.relu(h)

        _, *feat = h.shape

        h = jnp.reshape(h, (N, T, *feat))

        h = self.flatten(h)

        outputs_sequence, states_sequence, initial_carry = self.ataagru(
            h, a, action_trace, reset, carry, is_target=is_target
        )

        outputs_sequence = jax.nn.relu(outputs_sequence)

        outputs_sequence = outputs_sequence + self.skip_connection(h)

        outputs_sequence = self.phi(outputs_sequence)

        # Return both the GRU outputs and hidden states across the entire sequence along with initial hidden state
        return outputs_sequence, states_sequence, initial_carry


class ForagerMAGRUNetReLU(hk.Module):
    def __init__(self, hidden: int, actions: int, learn_initial_h=True, name: str = ""):
        super().__init__(name=name)
        self.hidden_size = hidden
        self.number_of_actions = actions
        w_init = hk.initializers.Orthogonal(np.sqrt(2))

        self.conv = hk.Conv2D(16, 3, 2, w_init=w_init, name="phi")

        self.flatten = hk.Flatten(preserve_dims=2, name="flatten")

        self.skip_connection = hk.Linear(
            self.hidden_size, w_init=w_init, name="skip_connection"
        )

        self.magru = MAGRU(
            self.hidden_size,
            self.number_of_actions,
            learn_initial_h=learn_initial_h,
            name="gru",
        )

        self.phi = hk.Flatten(preserve_dims=2, name="phi")

    def __call__(
        self,
        x: jnp.ndarray,
        a: Optional[jnp.ndarray] = None,
        reset: Optional[jnp.ndarray] = None,
        carry: Optional[jnp.ndarray] = None,
        is_target=False,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Args:
          x: Input tensor with shape [N, T, ...]
          a: Action tensor with shape [N, T]
          reset: Optional binary flag sequence with shape [N, T] indicating when to reset the GRU state.
                 For example, at episode boundaries.
          carry: The initial hidden state for RNN.

        Returns:
          outputs_sequence: Representation vectors sequence.
          states_sequence: The hidden states sequence.
        """
        # Add temporal dimension if given a single slice
        if len(x.shape) < 5:
            x = x[:, None]

        N, T, *feat = x.shape

        # Use No-Op action -1 to populate a if None
        if a is None:
            a = jnp.full((N, T), jnp.int32(-1))
        if len(a.shape) < 2:
            a = jnp.broadcast_to(a, (N, T))

        x = jnp.reshape(x, (N * T, *feat))

        h = self.conv(x)
        h = jax.nn.relu(h)

        _, *feat = h.shape

        h = jnp.reshape(h, (N, T, *feat))

        h = self.flatten(h)

        outputs_sequence, states_sequence, initial_carry = self.magru(
            h, a, reset, carry, is_target=is_target
        )

        outputs_sequence = jax.nn.relu(outputs_sequence)

        outputs_sequence = outputs_sequence + self.skip_connection(h)

        outputs_sequence = self.phi(outputs_sequence)

        # Return both the GRU outputs and hidden states across the entire sequence along with initial hidden state
        return outputs_sequence, states_sequence, initial_carry


class ForagerNet(hk.Module):
    def __init__(
        self,
        hidden: int,
        d_hidden: Optional[int] = None,
        scalars: int = 0,
        layers: Optional[int] = None,
        pre_core_layers: int = 0,
        core_layers: int = 0,
        post_core_layers: int = 0,
        use_layernorm=False,
        preactivation_layer_norm: bool = True,
        balanced=False,
        name: str = "",
        conv: str = "Conv2D",
        activation: Optional[str] = "relu",
        coord: bool = False,
        **kwargs,
    ):
        super().__init__(name=name)
        self.hidden = hidden
        self.d_hidden = d_hidden if d_hidden is not None else hidden
        self.scalars = scalars
        self.pre_core_layers = pre_core_layers
        self.core_layers = core_layers
        self.post_core_layers = post_core_layers
        self.layers = (
            layers
            if layers is not None
            else pre_core_layers + core_layers + post_core_layers
        )
        self.use_layernorm = use_layernorm
        self.preactivation_layer_norm = preactivation_layer_norm
        self.balanced = balanced
        self.coord_conv = coord
        if activation == "crelu":
            self.activation_fn = hku.crelu
        else:
            self.activation_fn = jax.nn.relu
        w_init = hk.initializers.Orthogonal(np.sqrt(2))

        conv_layers = []
        if conv == "PConv2D" or conv == "PConv2DConv2D":
            conv_layers.append(hk.Conv2D(16, 1, 1, w_init=w_init, name="phi"))
            if self.use_layernorm:
                conv_layers.append(
                    hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
                )
            conv_layers.append(self.activation_fn)
        if self.coord_conv:
            conv_layers.append(self._add_coord_channels)
        if conv == "PConv2DConv2D" or conv == "Conv2D":
            conv_layers.append(hk.Conv2D(16, 3, 1, w_init=w_init, name="phi"))
            if self.use_layernorm:
                conv_layers.append(
                    hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
                )
            conv_layers.append(self.activation_fn)

        self.conv = hk.Sequential(conv_layers)

        self.flatten = hk.Flatten(preserve_dims=1, name="flatten")

        # Balanced mode: project vision and scalars to equal-sized embeddings
        if self.balanced:
            vision_proj = [hk.Linear(self.hidden, w_init=w_init, name="vision_proj")]
            if use_layernorm:
                vision_proj.append(
                    hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
                )
            vision_proj.append(self.activation_fn)
            self.vision_proj = hk.Sequential(vision_proj)

            if self.scalars > 0:
                scalars_proj = [hk.Linear(self.hidden, w_init=w_init, name="scalars_proj")]
                if use_layernorm:
                    scalars_proj.append(
                        hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
                    )
                scalars_proj.append(self.activation_fn)
                self.scalars_proj = hk.Sequential(scalars_proj)

        def make_mlp(widths):
            mlp_layers = []
            for width in widths:
                mlp_layers.append(hk.Linear(width, w_init=w_init))
                if use_layernorm and self.preactivation_layer_norm:
                    mlp_layers.append(
                        hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
                    )
                mlp_layers.append(self.activation_fn)
                if use_layernorm and not self.preactivation_layer_norm:
                    mlp_layers.append(
                        hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
                    )
            return hku.accumulatingSequence(mlp_layers)

        if layers is not None:
            self.core_mlp = make_mlp([self.hidden] * layers)
        else:
            if self.pre_core_layers > 0:
                self.pre_core_mlp = make_mlp([self.hidden] * self.pre_core_layers)
            if self.core_layers > 0:
                self.core_mlp = make_mlp([self.d_hidden] * self.core_layers)
            if self.post_core_layers > 0:
                self.post_core_mlp = make_mlp([self.hidden] * self.post_core_layers)

        self.phi = hk.Flatten(preserve_dims=1, name="phi")

    @staticmethod
    def _add_coord_channels(x: jnp.ndarray) -> jnp.ndarray:
        """Append normalised (x, y) coordinate channels to image tensor.

        Works for shapes (..., H, W, C).  Coordinates are in [-1, 1].
        """
        *batch, h, w, _c = x.shape
        # Row coords (y) and column coords (x), normalised to [-1, 1]
        y_coords = jnp.linspace(-1.0, 1.0, h)[:, None]          # (H, 1)
        x_coords = jnp.linspace(-1.0, 1.0, w)[None, :]          # (1, W)
        y_grid = jnp.broadcast_to(y_coords, (h, w))[..., None]   # (H, W, 1)
        x_grid = jnp.broadcast_to(x_coords, (h, w))[..., None]   # (H, W, 1)
        coords = jnp.concatenate([x_grid, y_grid], axis=-1)      # (H, W, 2)
        # Broadcast across batch dims
        for _ in batch:
            coords = coords[None, ...]
        coords = jnp.broadcast_to(coords, (*batch, h, w, 2))
        return jnp.concatenate([x, coords], axis=-1)

    def __call__(
        self,
        x: jnp.ndarray,
        scalars: Optional[jnp.ndarray] = None,
        **kwargs,
    ) -> hku.AccumulatedOutput:
        h = self.conv(x)

        h = self.flatten(h)

        if self.balanced:
            h = self.vision_proj(h)

        mlp_acts: Dict[str, jnp.ndarray] = {}

        if self.pre_core_layers > 0 and hasattr(self, "pre_core_mlp"):
            pre_out = self.pre_core_mlp(h)
            h = pre_out.out
            for k, v in pre_out.activations.items():
                mlp_acts[f"pre_core/{k}"] = v

        if self.scalars > 0:
            if self.balanced:
                if scalars is not None:
                    scalars = self.scalars_proj(scalars)
                else:
                    scalars = jnp.zeros(x.shape[:-3] + (self.hidden,))
            elif scalars is None:
                scalars = jnp.zeros(x.shape[:-3] + (self.scalars,))
            h = jnp.concatenate([h, scalars], axis=-1)

        if hasattr(self, "core_mlp"):
            core_out = self.core_mlp(h)
            h = core_out.out
            prefix = (
                ""
                if (self.pre_core_layers == 0 and self.post_core_layers == 0)
                else "core/"
            )
            for k, v in core_out.activations.items():
                mlp_acts[f"{prefix}{k}"] = v

        if self.post_core_layers > 0 and hasattr(self, "post_core_mlp"):
            post_out = self.post_core_mlp(h)
            h = post_out.out
            for k, v in post_out.activations.items():
                mlp_acts[f"post_core/{k}"] = v

        out = self.phi(h)

        return hku.AccumulatedOutput(
            activations={self.name: out, **mlp_acts}, out=out
        )


def make_standalone_initializer(hk_initializer) -> Callable:
    """Convert a Haiku initializer to a standalone function.

    Haiku initializers expect to be called within a transform context.
    This wrapper creates a function that can be called with (key, shape, dtype).
    """

    def standalone_init(key, shape, dtype):
        def _inner():
            return hk_initializer(shape, dtype)

        transformed = hk.transform(_inner)
        return transformed.apply({}, key)

    return standalone_init
