Facilitating New Backend Integration by PrivateUse1
===================================================

In this tutorial we will walk through some necessary steps to integrate a new backend
living outside ``pytorch/pytorch`` repo by ``PrivateUse1``. Note that this tutorial assumes that
you already have a basic understanding of PyTorch.

.. note::

   This tutorial only touches the part related to the privateuse1 mechanism that facilitates the integration of new devices,
   Other public parts will not be involved (if you want to know the whole process of new device integration, you can refer
   to `Ascend NPU <https://gitee.com/ascend/pytorch>`_). Meanwhile, all the modules involved in this tutorial are not required,
   you can choose the one that is helpful to you according to your actual needs.

Why is it PrivateUse1?
----------------------

PyTorch provides three reserved dispatch keys (and their corresponding Autograd keys)
for prototyping out-of-tree backend extensions. After prototype verification is passed,
a private key can be applied for new backend.

* PrivateUse1/AutogradPrivateUse1
* PrivateUse2/AutogradPrivateUse2
* PrivateUse3/AutogradPrivateUse3

However, with the rapid development of Pytorch, more and more hardware manufacturers try to
integrate their backends into Pytorch, however, this will cause some problems:

* Every new backend integration involves a lot of file modification
* There is currently a hard limit on the number of Dispatch Keys
* These out-of-tree backends are rarely used at the same time

In view of the above reasons, the community began to recommend new hardware to be integrated
into the community through the public key(PrivateUse1).

However, the previous PrivateUse1 does not have the ability to integrate with new backends,
because PrivateUse1 lacks related support in some modules, such as Storage, AMP, Distributed, etc.

With the arrival of Pytorch 2.1.0, a series of optimizations and enhancements have been made
for PrivateUse1 in terms of new backend integration, and it is now possible to support the integration
of new devices rapidly and friendly (of course, there is still a lack of support on some modules,
but will be supported in the future)

Register kernels for the new backend
------------------------------------

The new backend may have some high-performance implementations of operator, which can be registered to the dispatcher
by ``TORCH_LIBRARY_IMPL`` API described in `Registering a Dispatched Operator in C++ <dispatcher>`_, it mainly involves
several situations.

1. Register all the forward operators supported by the new backend to the dispatcher and register the fallback
   at the same time, so that when some operators are not supported by the new backend, which can fallback to the
   CPU for execution to ensure availability of the functionality.

.. code-block:: cpp

  at::Tensor wrapper_Custom_Tensor_add(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
    // Implementation of add kernel in new backend
    ...
  }

  TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    ...
    m.impl("add.Tensor", TORCH_FN(wrapper_Custom_Tensor_add));
    ...
  }

  void custom_cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
    // Add some hints about new devices that do not support and need to fall back to cpu
    at::native::cpu_fallback(op, stack);
  }

  TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&custom_cpu_fallback>());
  }

2. Register kernels from ``torch::autograd::Function`` to the dispatcher by ``AutogradPrivateUse1``, if it is necessary for
   new backend to override ``PyTorch Autograd layer``, the dispatcher and autograd system will automatically call the forward and
   backward implementations of these operators.

.. code-block:: cpp

  class CumtomSeluFunction : public torch::autograd::Function<CumtomSeluFunction> {
    // Implementation of selu kernel in new backend
  }

  at::Tensor wrapper_AutogradCumstom__selu(const at::Tensor & self) {
    return CumtomSeluFunction::apply(self);
  }

  TORCH_LIBRARY_IMPL(aten, AutogradPrivateUse1, m) {
    ...
    m.impl("selu", TORCH_FN(wrapper_AutogradCustom__selu));
    ...
  }

3. Register kernels which want to support `automatic mixed precision (AMP) <https://pytorch.org/docs/stable/amp.html>`_ and
   fallback mechanism to the dispatcher by ``AutocastPrivateUse1``, the autocast system will automatically call these kernels when needed.

.. code-block:: cpp

  TORCH_LIBRARY_IMPL(aten, AutocastPrivateUse1, m) {
    ...
    KERNEL_PRIVATEUSEONE(<operator>, <policy>)
    ...
  }

  TORCH_LIBRARY_IMPL(_, AutocastPrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFallthrough());
  }

What needs to be added is that if new backend want to support AMP, which need to register a new ``BackendModule`` by
``torch._register_device_module("backend_name", BackendModule)``, and the ``BackendModule`` needs to have the following APIs:

* ``get_amp_supported_dtype() -> List[torch.dtype]``
    get the supported dtypes on new backend in AMP, which maybe supports one more dtype.
* ``is_autocast_enabled() -> bool``
    check the AMP is enabled or not on new backend.
* ``get_autocast_dtype() -> torch.dtype``
    get the supported dtype on new backend in AMP, which is set by ``set_autocast_dtype`` or the
    default dtype, and the default dtype is ``torch.float16``.
* ``set_autocast_enabled(bool) -> None``
    enable the AMP or not on new backend.
* ``set_autocast_dtype(dtype) -> None``
    set the supported dtype on new backend in AMP, and the dtype be contained in the dtypes got
    from ``get_amp_supported_dtype``.

Register generator for the new backend
--------------------------------------

It is necessary to support generators corresponding to new devices. Currently, PrivateUse1 can dynamically
register custom generators, which are mainly divided into the following steps.

1. Inherit the ``GeneratorImpl`` class to implement the generator class corresponding to the new backend,
   and implement various general methods.
2. Define a new backend ``builder`` with a single parameter: ``device index``.
3. Call ``REGISTER_GENERATOR_PRIVATEUSE1`` macro to complete dynamic registration.

.. code-block:: cpp

  struct CustomGeneratorImpl : public c10::GeneratorImpl {
    // Implementation of generator in new backend
  }

  at::Generator make_custom_generator(c10::DeviceIndex device_index) {
    return at::make_generator<CustomGeneratorImpl>(device_index);
  }

  REGISTER_GENERATOR_PRIVATEUSE1(make_cumstom_generator)

Register device guard for the new backend
--------------------------------------

Pytorch provides functionalities related to device, stream and event switching via DeviceGuard.
This function is also applicable to PrivateUse1 Key.

1. Inherit the ``DeviceGuardImplInterface`` class to implement the various general methods corresponding to the new backend.
2. Call ``C10_REGISTER_GUARD_IMPL`` macro to complete dynamic registration.

.. code-block:: cpp

  struct CustomGuardImpl final : public c10::impl::DeviceGuardImplInterface {
    // Implementation of guard in new backend
  }

  C10_REGISTER_GUARD_IMPL(PrivateUse1, CustomGuardImpl);

Rename privateuse1 to a custom name for the new backend
-------------------------------------------------------

PrivateUse1 Key is the internal mechanism of the new backend integrated into PyTorch. For users, compared with ``PrivateUse1``,
the custom name strongly related to the new backend should be more friendly.

For example, taking the ``Ascend NPU`` as an example, the first usage will be more user-friendly.

.. code-block:: python

  torch.rand([2,2],device='npu:0')
  torch.rand([2,2],device='privateuse1:0')

Now, PyTorch provides a new C++/Python API for the self-named "PrivateUse1" backend, which is very simple to use.

For C++:

.. code-block:: cpp

  c10::register_privateuse1_backend("custom_name")

For Python:

.. code-block:: python

  torch.rename_privateuse1_backend("custom_name")

Register serialization/deserialization functions for new backend metadata
-------------------------------------------------------------------------

PyTorch is currently able to dynamically register serialization/deserialization functions to support the serialization and deserialization
of new backend additional metadata named ``backend_meta_`` in clas ``TensorImpl.ExtraMeta``. You can refer to the following steps:

1. Inherit the ``BackendMeta`` class to implement ``CustomBackendMetadata`` corresponding to the new backend and
   various fields of new backend can be customized in the class.
2. Implement the serialization and deserialization functions of the new backend, the function signatures are 
   ``void(const at::Tensor&, std::unordered_map<std::string, bool>&)``
3. Call ``TensorBackendMetaRegistry`` macro to complete dynamic registration.

.. code-block:: cpp

  struct CustomBackendMetadata : public c10::BackendMeta {
    // Implementation of backend metadata in new backend
  }

  void for_serialization(const at::Tensor& t, std::unordered_map<std::string, bool>& m) {
    // Implementation of serialization
  }

  void for_deserialization(const at::Tensor& t, std::unordered_map<std::string, bool>& m) {
    // Implementation of deserialization
  }

  TensorBackendMetaRegistry(c10::DeviceType::PrivateUse1, &for_serialization, &for_deserialization);

Generate methods and properties related to the new backend
------------------------------------

After :ref:`rename privateuse1 to custome name <Rename privateuse1 to a custom name for the new backend>`, automatically generate properties
and methods related to the new backend name in the ``Tensor, nn, Storage`` modules for the new backend.

Examples are as follows:

.. code-block:: python

  torch.rename_privateuse1_backend("npu")
  unsupported_dtype = [torch.quint8]
  torch.utils.generate_methods_for_privateuse1_backend(for_tensor=True, for_module=True, for_storage=True, unsupported_dtype=unsupported_dtype)

Then, you can use the following methods and properties.

.. code-block:: python

  torch.Tensor.npu()
  torch.Tensor.is_npu
  torch.Storage.npu()
  torch.Storage.is_npu
  ...
