# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg31::tensorflow.compat.v1.name_scope
# name: tensorflow_primitive
# summary: Uses tensorflow.compat.v1.name_scope across 5 repos
# anchor_symbols: ['tensorflow.compat.v1.name_scope']
# observed in 5 repos: ['donnemartin__data-science-ipython-notebooks', 'lazyprogrammer__machine_learning_examples', 'piskvorky__gensim', 'tflearn__tflearn', 'vaexio__vaex']...

# --- from tflearn__tflearn::tflearn/distances.py::euclidean ---
def euclidean(a, b):
    return tf.sqrt(tf.reduce_sum(tf.square(a - b),
                                 reduction_indices=0))

# --- from donnemartin__data-science-ipython-notebooks::deep-learning/tensor-flow-examples/input_data.py::_read32 ---
def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)

# --- from lazyprogrammer__machine_learning_examples::tensorflow/input_data.py::_read32 ---
def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)

# --- from tflearn__tflearn::tflearn/datasets/fashion_mnist.py::_read32 ---
def _read32(bytestream):
    try:
        dt = numpy.dtype(numpy.uint32).newbyteorder('>')
        return numpy.frombuffer(bytestream.read(), dtype=dt)[0]
    except Exception:
        dt = numpy.dtype(numpy.uint32).newbyteorder('>')
        return numpy.frombuffer(bytestream.read(), dtype=dt)

# --- from donnemartin__data-science-ipython-notebooks::deep-learning/tensor-flow-examples/input_data.py::extract_labels ---
def extract_labels(filename, one_hot=False):
  """Extract the labels into a 1D uint8 numpy array [index]."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError(
          'Invalid magic number %d in MNIST label file: %s' %
          (magic, filename))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    if one_hot:
      return dense_to_one_hot(labels)
    return labels

# --- from lazyprogrammer__machine_learning_examples::tensorflow/input_data.py::extract_labels ---
def extract_labels(filename, one_hot=False):
  """Extract the labels into a 1D uint8 numpy array [index]."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError(
          'Invalid magic number %d in MNIST label file: %s' %
          (magic, filename))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    if one_hot:
      return dense_to_one_hot(labels)
    return labels

# --- from piskvorky__gensim::gensim/models/keyedvectors.py::_add_bytes_to_kv ---
def _add_bytes_to_kv(kv, counts, chunk, vocab_size, vector_size, datatype, unicode_errors, encoding):
    start = 0
    processed_words = 0
    bytes_per_vector = vector_size * dtype(REAL).itemsize
    max_words = vocab_size - kv.next_index  # don't read more than kv preallocated to hold
    assert max_words > 0
    for _ in range(max_words):
        i_space = chunk.find(b' ', start)
        i_vector = i_space + 1

        if i_space == -1 or (len(chunk) - i_vector) < bytes_per_vector:
            break

        word = chunk[start:i_space].decode(encoding, errors=unicode_errors)
        # Some binary files are reported to have obsolete new line in the beginning of word, remove it
        word = word.lstrip('\n')
        vector = frombuffer(chunk, offset=i_vector, count=vector_size, dtype=REAL).astype(datatype)
        _add_word_to_kv(kv, counts, word, vector, vocab_size)
        start = i_vector + bytes_per_vector
        processed_words += 1

    return processed_words, chunk[start:]

# --- from vaexio__vaex::packages/vaex-core/vaex/metal.py::FunctionSerializableMetal.compile.wrapper.getbuf ---
def getbuf(name, value=None, dtype=np.dtype("float32"), N=None):
                buf = getattr(storage, name, None)
                if value is not None:
                    N = len(value)
                    dtype = value.dtype
                if dtype.name == "float64":
                    warnings.warn("Casting input argument from float64 to float32 since Metal does not support float64")
                    dtype = np.dtype("float32")
                nbytes = N * dtype.itemsize
                if buf is not None and buf.length() != nbytes:
                    # doesn't match size, create a new one
                    buf = None
                # create a buffer
                if buf is None:
                    buf = self.device.newBufferWithLength_options_(nbytes, 0)
                    setattr(storage, name, buf)
                # copy data to buffer
                if value is not None:
                    mv = buf.contents().as_buffer(buf.length())
                    buf_as_numpy = np.frombuffer(mv, dtype=dtype)
                    buf_as_numpy[:] = value.astype(dtype, copy=False)
                return buf

# --- from vaexio__vaex::packages/vaex-core/vaex/metal.py::FunctionSerializableMetal.compile.wrapper ---
def wrapper(*args):
            args = [vaex.array_types.to_numpy(ar) for ar in args]
            def getbuf(name, value=None, dtype=np.dtype("float32"), N=None):
                buf = getattr(storage, name, None)
                if value is not None:
                    N = len(value)
                    dtype = value.dtype
                if dtype.name == "float64":
                    warnings.warn("Casting input argument from float64 to float32 since Metal does not support float64")
                    dtype = np.dtype("float32")
                nbytes = N * dtype.itemsize
                if buf is not None and buf.length() != nbytes:
                    # doesn't match size, create a new one
                    buf = None
                # create a buffer
                if buf is None:
                    buf = self.device.newBufferWithLength_options_(nbytes, 0)
                    setattr(storage, name, buf)
                # copy data to buffer
                if value is not None:
                    mv = buf.contents().as_buffer(buf.length())
                    buf_as_numpy = np.frombuffer(mv, dtype=dtype)
                    buf_as_numpy[:] = value.astype(dtype, copy=False)
                return buf
            input_buffers = [getbuf(name, chunk) for name, chunk in zip(self.arguments, args)]
            output_buffer = getbuf('vaex_output', N=len(args[0]), dtype=dtype_out)
            buffers = input_buffers + [output_buffer]
            command_buffer = command_queue.commandBuffer()
            encoder = command_buffer.computeCommandEncoder()
            encoder.setComputePipelineState_(state)
            for i, buf in enumerate(buffers):
                encoder.setBuffer_offset_atIndex_(buf, 0, i)
            nitems = len(args[0])
            tpgrid = Metal.MTLSize(width=nitems, height=1, depth=1)
            # state.threadExecutionWidth() == 32 on M1 max
            # state.maxTotalThreadsPerThreadgroup() == 1024 on M1 max
            tptgroup = Metal.MTLSize(width=state.threadExecutionWidth(), height=state.maxTotalThreadsPerThreadgroup()//state.threadExecutionWidth(), depth=1)
            # this is simpler, and gives the same performance
            # tptgroup = Metal.MTLSize(width=1, height=1, depth=1)
            encoder.dispatchThreads_threadsPerThreadgroup_(tpgrid, tptgroup)
            encoder.endEncoding()
            command_buffer.commit()
            command_buffer.waitUntilCompleted()

            output_buffer_py = output_buffer.contents().as_buffer(output_buffer.length())
             # do we needs .copy() ?
            result = np.frombuffer(output_buffer_py, dtype=dtype_out)
            return result
