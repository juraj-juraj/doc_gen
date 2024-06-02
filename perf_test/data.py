def load(file_path, parse_line_fn):

    vocabulary = []
    embeddings = []
    embeddings_dim = None
    for line in tf.gfile.GFile(file_path):
        token, embedding = parse_line_fn(line)
        if not embeddings_dim:
            embeddings_dim = len(embedding)
        elif embeddings_dim != len(embedding):
            raise ValueError(
                "Inconsistent embedding dimension detected, %d != %d for token %s",
                embeddings_dim,
                len(embedding),
                token,
            )

        vocabulary.append(token)
        embeddings.append(embedding)

    return vocabulary, np.array(embeddings)


def _round_hex(q, r):

    x = q
    z = r
    y = -x - z

    rx = np.round(x)
    ry = np.round(y)
    rz = np.round(z)

    dx = np.abs(rx - x)
    dy = np.abs(ry - y)
    dz = np.abs(rz - z)

    cond = (dx > dy) & (dx > dz)
    q = np.where(cond, -(ry + rz), rx)
    r = np.where(~cond & ~(dy > dz), -(rx + ry), rz)

    return q.astype(int), r.astype(int)


def _inter_manager_operations(self, other, how_to_join, func):

    reindexed_self, reindexed_other_list, joined_index = self.copartition(0, other, how_to_join, False)
    # unwrap list returned by `copartition`.
    reindexed_other = reindexed_other_list[0]
    new_columns = self._join_index_objects(0, other.columns, how_to_join, sort=False)
    # THere is an interesting serialization anomaly that happens if we do
    # not use the columns in `inter_data_op_builder` from here (e.g. if we
    # pass them in). Passing them in can cause problems, so we will just
    # use them from here.
    self_cols = self.columns
    other_cols = other.columns

    def inter_data_op_builder(left, right, func):
        left.columns = self_cols
        right.columns = other_cols
        # We reset here to make sure that the internal indexes match. We aligned
        # them in the previous step, so this step is to prevent mismatches.
        left.index = pandas.RangeIndex(len(left.index))
        right.index = pandas.RangeIndex(len(right.index))
        result = func(left, right)
        result.columns = pandas.RangeIndex(len(result.columns))
        return result

    new_data = reindexed_self.inter_data_operation(1, lambda l, r: inter_data_op_builder(l, r, func), reindexed_other)
    return self.__constructor__(new_data, joined_index, new_columns)


def list_files(root, suffix, prefix=False):

    root = os.path.expanduser(root)
    files = list(filter(lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix), os.listdir(root)))

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files


def make_innermost_getter(getter):

    @functools.wraps(getter)
    def _new_getter(kernel_results, *args, **kwargs):

        results_stack = []
        while hasattr(kernel_results, "inner_results"):
            results_stack.append(kernel_results)
            kernel_results = kernel_results.inner_results

        return getter(kernel_results, *args, **kwargs)

    return _new_getter


def _handle_unsupported(self, request: dict) -> dict:

    response = {
        "response": {
            "shouldEndSession": False,
            "outputSpeech": {"type": "PlainText", "text": self.config["unsupported_message"]},
            "card": {"type": "Simple", "content": self.config["unsupported_message"]},
        }
    }

    response = self._generate_response(response, request)

    return response


def inception_resnet_v2_arg_scope(weight_decay=0.00004, batch_norm_decay=0.9997, batch_norm_epsilon=0.001):

    # Set weight_decay for weights in conv2d and fully_connected layers.
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_regularizer=slim.l2_regularizer(weight_decay),
        biases_regularizer=slim.l2_regularizer(weight_decay),
    ):

        batch_norm_params = {
            "decay": batch_norm_decay,
            "epsilon": batch_norm_epsilon,
        }
        # Set activation_fn and parameters for batch_norm.
        with slim.arg_scope(
            [slim.conv2d], activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params
        ) as scope:
            return scope


def get_default_sess_config(mem_fraction=0.99):

    conf = tfv1.ConfigProto()

    conf.allow_soft_placement = True
    # conf.log_device_placement = True

    conf.intra_op_parallelism_threads = 1
    conf.inter_op_parallelism_threads = 0
    # TF benchmark use cpu_count() - gpu_thread_count(), e.g. 80 - 8 * 2
    # Didn't see much difference.

    conf.gpu_options.per_process_gpu_memory_fraction = mem_fraction

    # This hurt performance of large data pipeline:
    # https://github.com/tensorflow/benchmarks/commit/1528c46499cdcff669b5d7c006b7b971884ad0e6
    # conf.gpu_options.force_gpu_compatible = True

    conf.gpu_options.allow_growth = True

    # from tensorflow.core.protobuf import rewriter_config_pb2 as rwc
    # conf.graph_options.rewrite_options.memory_optimization = \
    #     rwc.RewriterConfig.HEURISTICS

    # May hurt performance?
    # conf.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    # conf.graph_options.place_pruned_graph = True
    return conf


def __init__(self, submission_id, submissions, storage_bucket):

    self.submission_id = submission_id
    self.storage_bucket = storage_bucket
    self.type = None
    self.submission = None
    if submission_id in submissions.attacks:
        self.type = TYPE_NONTARGETED
        self.submission = submissions.attacks[submission_id]
    elif submission_id in submissions.targeted_attacks:
        self.type = TYPE_TARGETED
        self.submission = submissions.targeted_attacks[submission_id]
    elif submission_id in submissions.defenses:
        self.type = TYPE_DEFENSE
        self.submission = submissions.defenses[submission_id]
    else:
        raise WorkerError('Submission with ID "{0}" not found'.format(submission_id))
    self.submission_dir = None
    self.extracted_submission_dir = None


def _move_dims_to_flat_end(x, axis, x_ndims, right_end=True):

    if not axis:
        return x

    # Suppose x.shape = [a, b, c, d]
    # Suppose axis = [1, 3]

    # other_dims = [0, 2] in example above.
    other_dims = sorted(set(range(x_ndims)).difference(axis))
    # x_permed.shape = [a, c, b, d]
    perm = other_dims + list(axis) if right_end else list(axis) + other_dims
    x_permed = tf.transpose(a=x, perm=perm)

    if x.shape.is_fully_defined():
        x_shape = x.shape.as_list()
        # other_shape = [a, c], end_shape = [b * d]
        other_shape = [x_shape[i] for i in other_dims]
        end_shape = [np.prod([x_shape[i] for i in axis])]
        full_shape = other_shape + end_shape if right_end else end_shape + other_shape
    else:
        other_shape = tf.gather(tf.shape(input=x), other_dims)
        full_shape = tf.concat([other_shape, [-1]] if right_end else [[-1], other_shape], axis=0)
    return tf.reshape(x_permed, shape=full_shape)


def add_skip_connection(self, u, v, connection_type):

    if connection_type not in [self.CONCAT_CONNECT, self.ADD_CONNECT]:
        raise ValueError(
            "connection_type should be NetworkDescriptor.CONCAT_CONNECT " "or NetworkDescriptor.ADD_CONNECT."
        )
    self.skip_connections.append((u, v, connection_type))


def reverse(path):

    if is_rooted(path) or ".." in path:
        from b2.manager import get_manager

        get_manager().errors()('reverse(path): path is either rooted or contains ".." in the path')
    if path == ".":
        return path
    path = os.path.normpath(path)
    # os.sep.join() is being used over os.path.join() due
    # to an extra '..' that is created by os.path.join()
    return os.sep.join(".." for t in path.split(os.sep))


def expand_docstring(**kwargs):

    def _fn_wrapped(fn):

        doc = inspect.cleandoc(fn.__doc__)
        for k, v in six.iteritems(kwargs):
            # Capture each ${k} reference to replace with v.
            # We wrap the replacement in a function so no backslash escapes
            # are processed.
            pattern = r"\$\{" + str(k) + r"\}"
            doc = re.sub(pattern, lambda match: v, doc)  # pylint: disable=cell-var-from-loop
        fn.__doc__ = doc
        return fn

    return _fn_wrapped


def _full_reduce(self, axis, map_func, reduce_func=None):

    if reduce_func is None:
        reduce_func = map_func

    mapped_parts = self.data.map_across_blocks(map_func)
    full_frame = mapped_parts.map_across_full_axis(axis, reduce_func)
    if axis == 0:
        columns = self.columns
        return self.__constructor__(full_frame, index=["__reduced__"], columns=columns)
    else:
        index = self.index
        return self.__constructor__(full_frame, index=index, columns=["__reduced__"])


def __init__(self, channel):

    self.GenerateAccessToken = channel.unary_unary(
        "/google.iam.credentials.v1.IAMCredentials/GenerateAccessToken",
        request_serializer=google_dot_iam_dot_credentials_dot_v1_dot_common__pb2.GenerateAccessTokenRequest.SerializeToString,
        response_deserializer=google_dot_iam_dot_credentials_dot_v1_dot_common__pb2.GenerateAccessTokenResponse.FromString,
    )
    self.GenerateIdToken = channel.unary_unary(
        "/google.iam.credentials.v1.IAMCredentials/GenerateIdToken",
        request_serializer=google_dot_iam_dot_credentials_dot_v1_dot_common__pb2.GenerateIdTokenRequest.SerializeToString,
        response_deserializer=google_dot_iam_dot_credentials_dot_v1_dot_common__pb2.GenerateIdTokenResponse.FromString,
    )
    self.SignBlob = channel.unary_unary(
        "/google.iam.credentials.v1.IAMCredentials/SignBlob",
        request_serializer=google_dot_iam_dot_credentials_dot_v1_dot_common__pb2.SignBlobRequest.SerializeToString,
        response_deserializer=google_dot_iam_dot_credentials_dot_v1_dot_common__pb2.SignBlobResponse.FromString,
    )
    self.SignJwt = channel.unary_unary(
        "/google.iam.credentials.v1.IAMCredentials/SignJwt",
        request_serializer=google_dot_iam_dot_credentials_dot_v1_dot_common__pb2.SignJwtRequest.SerializeToString,
        response_deserializer=google_dot_iam_dot_credentials_dot_v1_dot_common__pb2.SignJwtResponse.FromString,
    )
    self.GenerateIdentityBindingAccessToken = channel.unary_unary(
        "/google.iam.credentials.v1.IAMCredentials/GenerateIdentityBindingAccessToken",
        request_serializer=google_dot_iam_dot_credentials_dot_v1_dot_common__pb2.GenerateIdentityBindingAccessTokenRequest.SerializeToString,
        response_deserializer=google_dot_iam_dot_credentials_dot_v1_dot_common__pb2.GenerateIdentityBindingAccessTokenResponse.FromString,
    )


def _run_submission(self, metadata):

    if self._use_gpu:
        docker_binary = "nvidia-docker"
        container_name = metadata["container_gpu"]
    else:
        docker_binary = "docker"
        container_name = metadata["container"]
    if metadata["type"] == "defense":
        cmd = [
            docker_binary,
            "run",
            "--network=none",
            "-m=24g",
            "-v",
            "{0}:/input_images:ro".format(self._sample_input_dir),
            "-v",
            "{0}:/output_data".format(self._sample_output_dir),
            "-v",
            "{0}:/code".format(self._extracted_submission_dir),
            "-w",
            "/code",
            container_name,
            "./" + metadata["entry_point"],
            "/input_images",
            "/output_data/result.csv",
        ]
    else:
        epsilon = np.random.choice(ALLOWED_EPS)
        cmd = [
            docker_binary,
            "run",
            "--network=none",
            "-m=24g",
            "-v",
            "{0}:/input_images:ro".format(self._sample_input_dir),
            "-v",
            "{0}:/output_images".format(self._sample_output_dir),
            "-v",
            "{0}:/code".format(self._extracted_submission_dir),
            "-w",
            "/code",
            container_name,
            "./" + metadata["entry_point"],
            "/input_images",
            "/output_images",
            str(epsilon),
        ]
    logging.info("Command to run submission: %s", " ".join(cmd))
    return shell_call(cmd)


def run_bottleneck_on_image(
    sess, image_data, image_data_tensor, decoded_image_tensor, resized_input_tensor, bottleneck_tensor
):

    # First decode the JPEG image, resize it, and rescale the pixel values.
    resized_input_values = sess.run(decoded_image_tensor, {image_data_tensor: image_data})
    # Then run it through the recognition network.
    bottleneck_values = sess.run(bottleneck_tensor, {resized_input_tensor: resized_input_values})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


def __init__(self, float_value, places=7):

    self._float_value = float_value
    self._places = places


def get_checkpoint_path(model_path):

    if os.path.basename(model_path) == model_path:
        model_path = os.path.join(".", model_path)  # avoid #4921 and #6142
    if os.path.basename(model_path) == "checkpoint":
        assert tfv1.gfile.Exists(model_path), model_path
        model_path = tf.train.latest_checkpoint(os.path.dirname(model_path))
        # to be consistent with either v1 or v2

    # fix paths if provided a wrong one
    new_path = model_path
    if "00000-of-00001" in model_path:
        new_path = model_path.split(".data")[0]
    elif model_path.endswith(".index"):
        new_path = model_path.split(".index")[0]
    if new_path != model_path:
        logger.info("Checkpoint path {} is auto-corrected to {}.".format(model_path, new_path))
        model_path = new_path
    assert tfv1.gfile.Exists(model_path) or tfv1.gfile.Exists(model_path + ".index"), model_path
    return model_path


def __init__(self, submission_id, submissions, storage_bucket):

    super(AttackSubmission, self).__init__(submission_id, submissions, storage_bucket)
    if (self.type != TYPE_TARGETED) and (self.type != TYPE_NONTARGETED):
        raise WorkerError('Incorrect attack type for submission "{0}"'.format(submission_id))


def predict(self, train_x):

    k_trans = np.exp(-np.power(edit_distance_matrix(train_x, self._x), 2))
    y_mean = k_trans.dot(self._alpha_vector)  # Line 4 (y_mean = f_star)

    # compute inverse K_inv of K based on its Cholesky
    # decomposition L and its inverse L_inv
    l_inv = solve_triangular(self._l_matrix.T, np.eye(self._l_matrix.shape[0]))
    k_inv = l_inv.dot(l_inv.T)
    # Compute variance of predictive distribution
    y_var = np.ones(len(train_x), dtype=np.float)
    y_var -= np.einsum("ij,ij->i", np.dot(k_trans, k_inv), k_trans)

    # Check if any of the variances is negative because of
    # numerical issues. If yes: set the variance to 0.
    y_var_negative = y_var < 0
    if np.any(y_var_negative):
        y_var[y_var_negative] = 0.0
    return y_mean, np.sqrt(y_var)


def from_api_repr(cls, resource):

    config = cls(resource["sourceFormat"])
    for optcls in _OPTION_CLASSES:
        opts = resource.get(optcls._RESOURCE_NAME)
        if opts is not None:
            config._options = optcls.from_api_repr(opts)
            break
    config._properties = copy.deepcopy(resource)
    return config


def _enum_from_direction(direction):

    if isinstance(direction, int):
        return direction

    if direction == Query.ASCENDING:
        return enums.StructuredQuery.Direction.ASCENDING
    elif direction == Query.DESCENDING:
        return enums.StructuredQuery.Direction.DESCENDING
    else:
        msg = _BAD_DIR_STRING.format(direction, Query.ASCENDING, Query.DESCENDING)
        raise ValueError(msg)


def read_sql(cls, sql, con, index_col=None, **kwargs):

    if cls.read_sql_remote_task is None:
        return super(RayIO, cls).read_sql(sql, con, index_col=index_col, **kwargs)

    row_cnt_query = "SELECT COUNT(*) FROM ({})".format(sql)
    row_cnt = pandas.read_sql(row_cnt_query, con).squeeze()
    cols_names_df = pandas.read_sql("SELECT * FROM ({}) LIMIT 0".format(sql), con, index_col=index_col)
    cols_names = cols_names_df.columns
    num_parts = cls.frame_mgr_cls._compute_num_partitions()
    partition_ids = []
    index_ids = []
    limit = math.ceil(row_cnt / num_parts)
    for part in range(num_parts):
        offset = part * limit
        query = "SELECT * FROM ({}) LIMIT {} OFFSET {}".format(sql, limit, offset)
        partition_id = cls.read_sql_remote_task._remote(
            args=(num_parts, query, con, index_col, kwargs),
            num_return_vals=num_parts + 1,
        )
        partition_ids.append([cls.frame_partition_cls(obj) for obj in partition_id[:-1]])
        index_ids.append(partition_id[-1])

    if index_col is None:  # sum all lens returned from partitions
        index_lens = ray.get(index_ids)
        new_index = pandas.RangeIndex(sum(index_lens))
    else:  # concat index returned from partitions
        index_lst = [x for part_index in ray.get(index_ids) for x in part_index]
        new_index = pandas.Index(index_lst).set_names(index_col)

    new_query_compiler = cls.query_compiler_cls(cls.frame_mgr_cls(np.array(partition_ids)), new_index, cols_names)
    return new_query_compiler


def from_http_response(response):

    try:
        payload = response.json()
    except ValueError:
        payload = {"error": {"message": response.text or "unknown error"}}

    error_message = payload.get("error", {}).get("message", "unknown error")
    errors = payload.get("error", {}).get("errors", ())

    message = "{method} {url}: {error}".format(
        method=response.request.method, url=response.request.url, error=error_message
    )

    exception = from_http_status(response.status_code, message, errors=errors, response=response)
    return exception


def _log(self, utterance: Any, direction: str, dialog_id: Optional[Hashable] = None):

    if isinstance(utterance, str):
        pass
    elif isinstance(utterance, RichMessage):
        utterance = utterance.json()
    elif isinstance(utterance, (list, dict)):
        utterance = jsonify_data(utterance)
    else:
        utterance = str(utterance)

    dialog_id = str(dialog_id) if not isinstance(dialog_id, str) else dialog_id

    if self.log_file.tell() >= self.log_max_size * 1024:
        self.log_file.close()
        self.log_file = self._get_log_file()
    else:
        try:
            log_msg = {}
            log_msg["timestamp"] = self._get_timestamp_utc_str()
            log_msg["dialog_id"] = dialog_id
            log_msg["direction"] = direction
            log_msg["message"] = utterance
            log_str = json.dumps(log_msg, ensure_ascii=self.config["ensure_ascii"])
            self.log_file.write(f"{log_str}\n")
        except IOError:
            log.error("Failed to write dialog log.")


def _ConvertFieldValuePair(self, js, message):

    names = []
    message_descriptor = message.DESCRIPTOR
    fields_by_json_name = dict((f.json_name, f) for f in message_descriptor.fields)
    for name in js:
        try:
            field = fields_by_json_name.get(name, None)
            if not field:
                field = message_descriptor.fields_by_name.get(name, None)
            if not field:
                if self.ignore_unknown_fields:
                    continue
                raise ParseError(
                    'Message type "{0}" has no field named "{1}".'.format(message_descriptor.full_name, name)
                )
            if name in names:
                raise ParseError(
                    'Message type "{0}" should not have multiple '
                    '"{1}" fields.'.format(message.DESCRIPTOR.full_name, name)
                )
            names.append(name)
            # Check no other oneof field is parsed.
            if field.containing_oneof is not None:
                oneof_name = field.containing_oneof.name
                if oneof_name in names:
                    raise ParseError(
                        'Message type "{0}" should not have multiple '
                        '"{1}" oneof fields.'.format(message.DESCRIPTOR.full_name, oneof_name)
                    )
                names.append(oneof_name)

            value = js[name]
            if value is None:
                if (
                    field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_MESSAGE
                    and field.message_type.full_name == "google.protobuf.Value"
                ):
                    sub_message = getattr(message, field.name)
                    sub_message.null_value = 0
                else:
                    message.ClearField(field.name)
                continue

            # Parse field value.
            if _IsMapEntry(field):
                message.ClearField(field.name)
                self._ConvertMapFieldValue(value, message, field)
            elif field.label == descriptor.FieldDescriptor.LABEL_REPEATED:
                message.ClearField(field.name)
                if not isinstance(value, list):
                    raise ParseError("repeated field {0} must be in [] which is " "{1}.".format(name, value))
                if field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_MESSAGE:
                    # Repeated message field.
                    for item in value:
                        sub_message = getattr(message, field.name).add()
                        # None is a null_value in Value.
                        if item is None and sub_message.DESCRIPTOR.full_name != "google.protobuf.Value":
                            raise ParseError("null is not allowed to be used as an element" " in a repeated field.")
                        self.ConvertMessage(item, sub_message)
                else:
                    # Repeated scalar field.
                    for item in value:
                        if item is None:
                            raise ParseError("null is not allowed to be used as an element" " in a repeated field.")
                        getattr(message, field.name).append(_ConvertScalarFieldValue(item, field))
            elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_MESSAGE:
                sub_message = getattr(message, field.name)
                sub_message.SetInParent()
                self.ConvertMessage(value, sub_message)
            else:
                setattr(message, field.name, _ConvertScalarFieldValue(value, field))
        except ParseError as e:
            if field and field.containing_oneof is None:
                raise ParseError("Failed to parse {0} field: {1}".format(name, e))
            else:
                raise ParseError(str(e))
        except ValueError as e:
            raise ParseError("Failed to parse {0} field: {1}.".format(name, e))
        except TypeError as e:
            raise ParseError("Failed to parse {0} field: {1}.".format(name, e))


def __init__(self, config):

    super(ModelExporter, self).__init__()
    self.config = config


def call_fn(fn, args):

    if expand_as_args(args):
        return fn(*args)
    elif _expand_as_kwargs(args):
        return fn(**args)
    else:
        return fn(args)


def verify_signature(amazon_cert: crypto.X509, signature: str, request_body: bytes) -> bool:

    signature = base64.b64decode(signature)

    try:
        crypto.verify(amazon_cert, signature, request_body, "sha1")
        result = True
    except crypto.Error:
        result = False

    return result


def __getitem__(self, key):

    getitem = self._class_to_mock.__dict__.get("__getitem__", None)

    # Verify the class supports item assignment.
    if getitem is None:
        raise TypeError("unsubscriptable object")

    # If we are in replay mode then simply call the mock __getitem__ method.
    if self._replay_mode:
        return MockMethod("__getitem__", self._expected_calls_queue, self._replay_mode)(key)

    # Otherwise, create a mock method __getitem__.
    return self._CreateMockMethod("__getitem__")(key)


def inter_data_operation(self, axis, func, other):

    if axis:
        partitions = self.row_partitions
        other_partitions = other.row_partitions
    else:
        partitions = self.column_partitions
        other_partitions = other.column_partitions
    func = self.preprocess_func(func)
    result = np.array(
        [
            partitions[i].apply(
                func,
                num_splits=self._compute_num_partitions(),
                other_axis_partition=other_partitions[i],
            )
            for i in range(len(partitions))
        ]
    )
    return self.__constructor__(result) if axis else self.__constructor__(result.T)


def draw(vertexes, edges):

    # pylint: disable=too-many-locals
    # NOTE: coordinates might me negative, so we need to shift
    # everything to the positive plane before we actually draw it.
    Xs = []  # pylint: disable=invalid-name
    Ys = []  # pylint: disable=invalid-name

    sug = _build_sugiyama_layout(vertexes, edges)

    for vertex in sug.g.sV:
        # NOTE: moving boxes w/2 to the left
        Xs.append(vertex.view.xy[0] - vertex.view.w / 2.0)
        Xs.append(vertex.view.xy[0] + vertex.view.w / 2.0)
        Ys.append(vertex.view.xy[1])
        Ys.append(vertex.view.xy[1] + vertex.view.h)

    for edge in sug.g.sE:
        for x, y in edge.view._pts:  # pylint: disable=protected-access
            Xs.append(x)
            Ys.append(y)

    minx = min(Xs)
    miny = min(Ys)
    maxx = max(Xs)
    maxy = max(Ys)

    canvas_cols = int(math.ceil(math.ceil(maxx) - math.floor(minx))) + 1
    canvas_lines = int(round(maxy - miny))

    canvas = AsciiCanvas(canvas_cols, canvas_lines)

    # NOTE: first draw edges so that node boxes could overwrite them
    for edge in sug.g.sE:
        # pylint: disable=protected-access
        assert len(edge.view._pts) > 1
        for index in range(1, len(edge.view._pts)):
            start = edge.view._pts[index - 1]
            end = edge.view._pts[index]

            start_x = int(round(start[0] - minx))
            start_y = int(round(start[1] - miny))
            end_x = int(round(end[0] - minx))
            end_y = int(round(end[1] - miny))

            assert start_x >= 0
            assert start_y >= 0
            assert end_x >= 0
            assert end_y >= 0

            canvas.line(start_x, start_y, end_x, end_y, "*")

    for vertex in sug.g.sV:
        # NOTE: moving boxes w/2 to the left
        x = vertex.view.xy[0] - vertex.view.w / 2.0
        y = vertex.view.xy[1]

        canvas.box(
            int(round(x - minx)),
            int(round(y - miny)),
            vertex.view.w,
            vertex.view.h,
        )

        canvas.text(int(round(x - minx)) + 1, int(round(y - miny)) + 1, vertex.data)

    canvas.draw()


def _apply_func_to_list_of_partitions(self, func, partitions, **kwargs):

    preprocessed_func = self.preprocess_func(func)
    return [obj.apply(preprocessed_func, **kwargs) for obj in partitions]


def _validate_bn_layer(self, layer):

    if not isinstance(layer, tf.keras.layers.BatchNormalization) and not isinstance(
        layer, tf.compat.v1.layers.BatchNormalization
    ):
        raise ValueError("batchnorm_layer must be an instance of BatchNormalization layer.")
    if layer.renorm:
        raise ValueError("BatchNorm Bijector does not support renormalization.")
    if layer.virtual_batch_size:
        raise ValueError("BatchNorm Bijector does not support virtual batch sizes.")


def _trigger(self):

    self._completed.set()
    for callback in self._callbacks:
        callback(self)


def from_str(cls, label: str) -> int:

    label_norm = label.replace("1", "one").upper()
    if label_norm in cls.__members__:
        return DecayType[label_norm]
    else:
        raise NotImplementedError


def erfinv(x, name="erfinv"):

    with tf.name_scope(name):
        x = tf.convert_to_tensor(value=x, name="x")
        if dtype_util.as_numpy_dtype(x.dtype) not in [np.float32, np.float64]:
            raise TypeError(
                "x.dtype={} is not handled, see docstring for supported " "types.".format(dtype_util.name(x.dtype))
            )
        return ndtri((x + 1.0) / 2.0) / np.sqrt(2.0)


def PReLU(x, init=0.001, name="output"):

    init = tfv1.constant_initializer(init)
    alpha = tfv1.get_variable("alpha", [], initializer=init)
    x = (1 + alpha) * x + (1 - alpha) * tf.abs(x)
    ret = tf.multiply(x, 0.5, name=name)

    ret.variables = VariableHolder(alpha=alpha)
    return ret


def get_messages(module):

    answer = collections.OrderedDict()
    for name in dir(module):
        candidate = getattr(module, name)
        if inspect.isclass(candidate) and issubclass(candidate, message.Message):
            answer[name] = candidate
    return answer


def __init__(self, channel):

    self.CompleteQuery = channel.unary_unary(
        "/google.cloud.talent.v4beta1.Completion/CompleteQuery",
        request_serializer=google_dot_cloud_dot_talent__v4beta1_dot_proto_dot_completion__service__pb2.CompleteQueryRequest.SerializeToString,
        response_deserializer=google_dot_cloud_dot_talent__v4beta1_dot_proto_dot_completion__service__pb2.CompleteQueryResponse.FromString,
    )


def point(self, x, y, char):

    assert len(char) == 1
    assert x >= 0
    assert x < self.cols
    assert y >= 0
    assert y < self.lines

    self.canvas[y][x] = char


def _make_tags_vector(self, tags, bucket_length=None) -> np.ndarray:

    bucket_length = bucket_length or len(tags)
    answer = np.zeros(shape=(bucket_length,), dtype=np.int32)
    for i, tag in enumerate(tags):
        answer[i] = self.tags.tok2idx(tag)
    return answer


def _load_submissions_from_datastore_dir(self, dir_suffix, id_pattern):

    submissions = self._storage_client.list_blobs(prefix=os.path.join(self._round_name, dir_suffix))
    return {
        id_pattern.format(idx): SubmissionDescriptor(path=s, participant_id=participant_from_submission_path(s))
        for idx, s in enumerate(submissions)
    }


def replace(s, pattern, replacement):

    # the replacement string may contain invalid backreferences (like \1 or \g)
    # which will cause python's regex to blow up. Since this should emulate
    # the jam version exactly and the jam version didn't support
    # backreferences, this version shouldn't either. re.sub
    # allows replacement to be a callable; this is being used
    # to simply return the replacement string and avoid the hassle
    # of worrying about backreferences within the string.
    def _replacement(matchobj):
        return replacement

    return re.sub(pattern, _replacement, s)


def percentile(self, percent):

    # Sanity check: Any value over 100 should become 100.
    if percent >= 100:
        percent = 100

    # Determine the actual target number.
    target = len(self) - len(self) * (percent / 100)

    # Iterate over the values in reverse, dropping the target by the
    # number of times each value has been seen. When the target passes
    # 0, return the value we are currently viewing.
    for k in reversed(sorted(self._data.keys())):
        target -= self._data[k]
        if target < 0:
            return k

    # The only way to get here is if there was no data.
    # In this case, just return 10 seconds.
    return 10


def _load_dataset_clipping(self, dataset_dir, epsilon):

    self.dataset_max_clip = {}
    self.dataset_min_clip = {}
    self._dataset_image_count = 0
    for fname in os.listdir(dataset_dir):
        if not fname.endswith(".png"):
            continue
        image_id = fname[:-4]
        image = np.array(Image.open(os.path.join(dataset_dir, fname)).convert("RGB"))
        image = image.astype("int32")
        self._dataset_image_count += 1
        self.dataset_max_clip[image_id] = np.clip(image + epsilon, 0, 255).astype("uint8")
        self.dataset_min_clip[image_id] = np.clip(image - epsilon, 0, 255).astype("uint8")


def __init__(self, image, segments):

    self.image = image
    self.segments = segments
    self.intercept = {}
    self.local_exp = {}
    self.local_pred = None


def diff(self, periods=1, axis=0):

    axis = self._get_axis_number(axis)
    return self.__constructor__(query_compiler=self._query_compiler.diff(periods=periods, axis=axis))


def read_hdf(cls, path_or_buf, **kwargs):

    if cls.read_hdf_remote_task is None:
        return super(RayIO, cls).read_hdf(path_or_buf, **kwargs)

    format = cls._validate_hdf_format(path_or_buf=path_or_buf)

    if format is None:
        ErrorMessage.default_to_pandas(
            "File format seems to be `fixed`. For better distribution consider saving the file in `table` format. "
            "df.to_hdf(format=`table`)."
        )
        return cls.from_pandas(pandas.read_hdf(path_or_buf=path_or_buf, **kwargs))

    columns = kwargs.get("columns", None)
    if not columns:
        empty_pd_df = pandas.read_hdf(path_or_buf, start=0, stop=0)
        columns = empty_pd_df.columns

    num_partitions = cls.frame_mgr_cls._compute_num_partitions()
    num_splits = min(len(columns), num_partitions)
    # Each item in this list will be a list of column names of the original df
    column_splits = (
        len(columns) // num_partitions if len(columns) % num_partitions == 0 else len(columns) // num_partitions + 1
    )
    col_partitions = [columns[i : i + column_splits] for i in range(0, len(columns), column_splits)]
    blk_partitions = np.array(
        [
            cls.read_hdf_remote_task._remote(
                args=(path_or_buf, cols, num_splits, kwargs),
                num_return_vals=num_splits + 1,
            )
            for cols in col_partitions
        ]
    ).T
    remote_partitions = np.array([[cls.frame_partition_cls(obj) for obj in row] for row in blk_partitions[:-1]])
    index_len = ray.get(blk_partitions[-1][0])
    index = pandas.RangeIndex(index_len)
    new_query_compiler = cls.query_compiler_cls(cls.frame_mgr_cls(remote_partitions), index, columns)
    return new_query_compiler


def __init__(self, dct_type=2, validate_args=False, name="dct"):

    # TODO(b/115910664): Support other DCT types.
    if dct_type not in (2, 3):
        raise NotImplementedError("`type` must be one of 2 or 3")
    self._dct_type = dct_type
    super(DiscreteCosineTransform, self).__init__(
        forward_min_event_ndims=1,
        inverse_min_event_ndims=1,
        is_constant_jacobian=True,
        validate_args=validate_args,
        name=name,
    )


def summarize_neural_network_spec(mlmodel_spec):

    inputs = [(blob.name, _get_feature_description_summary(blob)) for blob in mlmodel_spec.description.input]
    outputs = [(blob.name, _get_feature_description_summary(blob)) for blob in mlmodel_spec.description.output]
    nn = None

    if mlmodel_spec.HasField("neuralNetwork"):
        nn = mlmodel_spec.neuralNetwork
    elif mlmodel_spec.HasField("neuralNetworkClassifier"):
        nn = mlmodel_spec.neuralNetworkClassifier
    elif mlmodel_spec.HasField("neuralNetworkRegressor"):
        nn = mlmodel_spec.neuralNetworkRegressor

    layers = [_summarize_network_layer_info(layer) for layer in nn.layers] if nn != None else None
    return (inputs, outputs, layers)


def same_dynamic_shape(a, b):

    a = tf.convert_to_tensor(value=a, name="a")
    b = tf.convert_to_tensor(value=b, name="b")

    # Here we can't just do tf.equal(a.shape, b.shape), since
    # static shape inference may break the equality comparison between
    # shape(a) and shape(b) in tf.equal.
    def all_shapes_equal():
        return tf.reduce_all(
            input_tensor=tf.equal(
                tf.concat([tf.shape(input=a), tf.shape(input=b)], 0),
                tf.concat([tf.shape(input=b), tf.shape(input=a)], 0),
            )
        )

    # One of the shapes isn't fully defined, so we need to use the dynamic
    # shape.
    return tf.cond(pred=tf.equal(tf.rank(a), tf.rank(b)), true_fn=all_shapes_equal, false_fn=lambda: tf.constant(False))


def build_backward_pass_step(get_transition_matrix_for_timestep):

    def backward_pass_step(state, filtered_parameters):

        (filtered_mean, filtered_cov, predicted_mean, predicted_cov) = filtered_parameters
        transition_matrix = get_transition_matrix_for_timestep(state.timestep)

        next_posterior_mean = state.backward_mean
        next_posterior_cov = state.backward_cov

        posterior_mean, posterior_cov = backward_smoothing_update(
            filtered_mean,
            filtered_cov,
            predicted_mean,
            predicted_cov,
            next_posterior_mean,
            next_posterior_cov,
            transition_matrix,
        )

        return BackwardPassState(backward_mean=posterior_mean, backward_cov=posterior_cov, timestep=state.timestep - 1)

    return backward_pass_step


def __init__(self, channel):

    self.BatchAnnotateImages = channel.unary_unary(
        "/google.cloud.vision.v1p4beta1.ImageAnnotator/BatchAnnotateImages",
        request_serializer=google_dot_cloud_dot_vision__v1p4beta1_dot_proto_dot_image__annotator__pb2.BatchAnnotateImagesRequest.SerializeToString,
        response_deserializer=google_dot_cloud_dot_vision__v1p4beta1_dot_proto_dot_image__annotator__pb2.BatchAnnotateImagesResponse.FromString,
    )
    self.BatchAnnotateFiles = channel.unary_unary(
        "/google.cloud.vision.v1p4beta1.ImageAnnotator/BatchAnnotateFiles",
        request_serializer=google_dot_cloud_dot_vision__v1p4beta1_dot_proto_dot_image__annotator__pb2.BatchAnnotateFilesRequest.SerializeToString,
        response_deserializer=google_dot_cloud_dot_vision__v1p4beta1_dot_proto_dot_image__annotator__pb2.BatchAnnotateFilesResponse.FromString,
    )
    self.AsyncBatchAnnotateImages = channel.unary_unary(
        "/google.cloud.vision.v1p4beta1.ImageAnnotator/AsyncBatchAnnotateImages",
        request_serializer=google_dot_cloud_dot_vision__v1p4beta1_dot_proto_dot_image__annotator__pb2.AsyncBatchAnnotateImagesRequest.SerializeToString,
        response_deserializer=google_dot_longrunning_dot_operations__pb2.Operation.FromString,
    )
    self.AsyncBatchAnnotateFiles = channel.unary_unary(
        "/google.cloud.vision.v1p4beta1.ImageAnnotator/AsyncBatchAnnotateFiles",
        request_serializer=google_dot_cloud_dot_vision__v1p4beta1_dot_proto_dot_image__annotator__pb2.AsyncBatchAnnotateFilesRequest.SerializeToString,
        response_deserializer=google_dot_longrunning_dot_operations__pb2.Operation.FromString,
    )


def _value_and_batch_jacobian(f, x):

    if tf.executing_eagerly():
        with tf.GradientTape() as tape:
            tape.watch(x)
            value = f(x)
        batch_jacobian = tape.batch_jacobian(value, x)
    else:
        value = f(x)
        batch_jacobian = gradients.batch_jacobian(value, x)
    return value, batch_jacobian


def _get_best(values: List[float], losses: List[float], max_loss_div: float = 0.9, min_val_div: float = 10.0) -> float:

    assert len(values) == len(losses), "lengths of values and losses should be equal"
    min_ind = np.argmin(losses)
    for i in range(min_ind - 1, 0, -1):
        if (losses[i] * max_loss_div > losses[min_ind]) or (values[i] * min_val_div < values[min_ind]):
            return values[i + 1]
    return values[min_ind] / min_val_div


def make_m_psd(self, original_nu, feed_dictionary):

    feed_dict = feed_dictionary.copy()
    _, min_eig_val_m = self.get_lanczos_eig(compute_m=True, feed_dict=feed_dict)

    lower_nu = original_nu
    upper_nu = original_nu
    num_iter = 0

    # Find an upper bound on nu
    while min_eig_val_m - TOL < 0 and num_iter < (MAX_BINARY_SEARCH_ITER / 2):
        num_iter += 1
        upper_nu *= NU_UPDATE_CONSTANT
        feed_dict.update({self.nu: upper_nu})
        _, min_eig_val_m = self.get_lanczos_eig(compute_m=True, feed_dict=feed_dict)

    final_nu = upper_nu

    # Perform binary search to find best value of nu
    while lower_nu <= upper_nu and num_iter < MAX_BINARY_SEARCH_ITER:
        num_iter += 1
        mid_nu = (lower_nu + upper_nu) / 2
        feed_dict.update({self.nu: mid_nu})
        _, min_eig_val_m = self.get_lanczos_eig(compute_m=True, feed_dict=feed_dict)
        if min_eig_val_m - TOL < 0:
            lower_nu = mid_nu
        else:
            upper_nu = mid_nu

    final_nu = upper_nu

    return final_nu


def tail(self, n):

    # See head for an explanation of the transposed behavior
    if n < 0:
        n = max(0, len(self.index) + n)
    if self._is_transposed:
        result = self.__constructor__(
            self.data.transpose().take(1, -n).transpose(),
            self.index[-n:],
            self.columns,
            self._dtype_cache,
        )
        result._is_transposed = True
    else:
        result = self.__constructor__(self.data.take(0, -n), self.index[-n:], self.columns, self._dtype_cache)
    return result


def crop_and_resize(image, boxes, box_ind, crop_size, pad_border=True):

    assert isinstance(crop_size, int), crop_size
    boxes = tf.stop_gradient(boxes)

    # TF's crop_and_resize produces zeros on border
    if pad_border:
        # this can be quite slow
        image = tf.pad(image, [[0, 0], [0, 0], [1, 1], [1, 1]], mode="SYMMETRIC")
        boxes = boxes + 1

    @under_name_scope()
    def transform_fpcoor_for_tf(boxes, image_shape, crop_shape):

        x0, y0, x1, y1 = tf.split(boxes, 4, axis=1)

        spacing_w = (x1 - x0) / tf.cast(crop_shape[1], tf.float32)
        spacing_h = (y1 - y0) / tf.cast(crop_shape[0], tf.float32)

        imshape = [tf.cast(image_shape[0] - 1, tf.float32), tf.cast(image_shape[1] - 1, tf.float32)]
        nx0 = (x0 + spacing_w / 2 - 0.5) / imshape[1]
        ny0 = (y0 + spacing_h / 2 - 0.5) / imshape[0]

        nw = spacing_w * tf.cast(crop_shape[1] - 1, tf.float32) / imshape[1]
        nh = spacing_h * tf.cast(crop_shape[0] - 1, tf.float32) / imshape[0]

        return tf.concat([ny0, nx0, ny0 + nh, nx0 + nw], axis=1)

    # Expand bbox to a minium size of 1
    # boxes_x1y1, boxes_x2y2 = tf.split(boxes, 2, axis=1)
    # boxes_wh = boxes_x2y2 - boxes_x1y1
    # boxes_center = tf.reshape((boxes_x2y2 + boxes_x1y1) * 0.5, [-1, 2])
    # boxes_newwh = tf.maximum(boxes_wh, 1.)
    # boxes_x1y1new = boxes_center - boxes_newwh * 0.5
    # boxes_x2y2new = boxes_center + boxes_newwh * 0.5
    # boxes = tf.concat([boxes_x1y1new, boxes_x2y2new], axis=1)

    image_shape = tf.shape(image)[2:]
    boxes = transform_fpcoor_for_tf(boxes, image_shape, [crop_size, crop_size])
    image = tf.transpose(image, [0, 2, 3, 1])  # nhwc
    ret = tf.image.crop_and_resize(image, boxes, tf.cast(box_ind, tf.int32), crop_size=[crop_size, crop_size])
    ret = tf.transpose(ret, [0, 3, 1, 2])  # ncss
    return ret


def __init__(self, dimensions, hidden_size):

    super(LearnableMultivariateNormalDiagCell, self).__init__()
    self.dimensions = dimensions
    self.hidden_size = hidden_size
    self.lstm_cell = tf.keras.layers.LSTMCell(hidden_size)
    self.output_layer = tf.keras.layers.Dense(2 * dimensions)


def inverse_removing(self, words_to_remove):

    mask = np.ones(self.as_np.shape[0], dtype="bool")
    mask[self.__get_idxs(words_to_remove)] = False
    if not self.bow:
        return "".join([self.as_list[i] if mask[i] else "UNKWORDZ" for i in range(mask.shape[0])])
    return "".join([self.as_list[v] for v in mask.nonzero()[0]])


def _kl_blockwise_blockwise(b0, b1, name=None):

    if len(b0.distributions) != len(b1.distributions):
        raise ValueError(
            "Can only compute KL divergence between Blockwise distributions with "
            "the same number of component distributions."
        )

    # We also need to check that the event shapes match for each one.
    b0_event_sizes = [_event_size(d) for d in b0.distributions]
    b1_event_sizes = [_event_size(d) for d in b1.distributions]

    assertions = []
    message = "Can only compute KL divergence between Blockwise distributions " "with the same pairwise event shapes."

    if all(isinstance(event_size, int) for event_size in b0_event_sizes) and all(
        isinstance(event_size, int) for event_size in b1_event_sizes
    ):
        if b0_event_sizes != b1_event_sizes:
            raise ValueError(message)
    else:
        if b0.validate_args or b1.validate_args:
            assertions.extend(
                assert_util.assert_equal(e1, e2, message=message)  # pylint: disable=g-complex-comprehension
                for e1, e2 in zip(b0_event_sizes, b1_event_sizes)
            )

    with tf.name_scope(name or "kl_blockwise_blockwise"):
        with tf.control_dependencies(assertions):
            return sum([kullback_leibler.kl_divergence(d1, d2) for d1, d2 in zip(b0.distributions, b1.distributions)])


def process_word(word: str, to_lower: bool = False, append_case: Optional[str] = None) -> Tuple[str]:

    if all(x.isupper() for x in word) and len(word) > 1:
        uppercase = "<ALL_UPPER>"
    elif word[0].isupper():
        uppercase = "<FIRST_UPPER>"
    else:
        uppercase = None
    if to_lower:
        word = word.lower()
    if word.isdigit():
        answer = ["<DIGIT>"]
    elif word.startswith("http://") or word.startswith("www."):
        answer = ["<HTTP>"]
    else:
        answer = list(word)
    if to_lower and uppercase is not None:
        if append_case == "first":
            answer = [uppercase] + answer
        elif append_case == "last":
            answer = answer + [uppercase]
    return tuple(answer)


def _from_any_pb(pb_type, any_pb):

    msg = pb_type()
    if not any_pb.Unpack(msg):
        raise TypeError("Could not convert {} to {}".format(any_pb.__class__.__name__, pb_type.__name__))

    return msg


def to_tensor(pic):

    if not (_is_pil_image(pic) or _is_numpy_image(pic)):
        raise TypeError("pic should be PIL Image or ndarray. Got {}".format(type(pic)))

    if isinstance(pic, np.ndarray):
        # handle numpy array
        if pic.ndim == 2:
            pic = pic[:, :, None]

        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img

    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return torch.from_numpy(nppic)

    # handle PIL Image
    if pic.mode == "I":
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == "I;16":
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    elif pic.mode == "F":
        img = torch.from_numpy(np.array(pic, np.float32, copy=False))
    elif pic.mode == "1":
        img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == "YCbCr":
        nchannel = 3
    elif pic.mode == "I;16":
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img


def CopyToProto(self, proto):

    if self.file is not None and self._serialized_start is not None and self._serialized_end is not None:
        proto.ParseFromString(self.file.serialized_pb[self._serialized_start : self._serialized_end])
    else:
        raise Error("Descriptor does not contain serialization.")


def make_innermost_setter(setter):

    @functools.wraps(setter)
    def _new_setter(kernel_results, *args, **kwargs):

        results_stack = []
        while hasattr(kernel_results, "inner_results"):
            results_stack.append(kernel_results)
            kernel_results = kernel_results.inner_results

        new_kernel_results = setter(kernel_results, *args, **kwargs)
        for outer_results in reversed(results_stack):
            new_kernel_results = outer_results._replace(inner_results=new_kernel_results)

        return new_kernel_results

    return _new_setter


def build_output_map(protomap, get_tensor_by_name):

    def get_output_from_tensor_info(tensor_info):
        encoding = tensor_info.WhichOneof("encoding")
        if encoding == "name":
            return get_tensor_by_name(tensor_info.name)
        elif encoding == "coo_sparse":
            return tf.SparseTensor(
                get_tensor_by_name(tensor_info.coo_sparse.indices_tensor_name),
                get_tensor_by_name(tensor_info.coo_sparse.values_tensor_name),
                get_tensor_by_name(tensor_info.coo_sparse.dense_shape_tensor_name),
            )
        else:
            raise ValueError("Invalid TensorInfo.encoding: %s" % encoding)

    return {key: get_output_from_tensor_info(tensor_info) for key, tensor_info in protomap.items()}


def get_diff_trees(self, a_ref, b_ref=None):

    diff_dct = {DIFF_EQUAL: False}
    trees, commit_refs = self._get_diff_trees(a_ref, b_ref)
    diff_dct[DIFF_A_REF] = commit_refs[0]
    diff_dct[DIFF_B_REF] = commit_refs[1]
    if commit_refs[0] == commit_refs[1]:
        diff_dct[DIFF_EQUAL] = True
        return diff_dct
    diff_dct[DIFF_A_TREE] = trees[DIFF_A_TREE]
    diff_dct[DIFF_B_TREE] = trees[DIFF_B_TREE]
    return diff_dct


def _fill_from_default(self, default_job_config):

    if self._job_type != default_job_config._job_type:
        raise TypeError(
            "attempted to merge two incompatible job types: "
            + repr(self._job_type)
            + ", "
            + repr(default_job_config._job_type)
        )

    new_job_config = self.__class__()

    default_job_properties = copy.deepcopy(default_job_config._properties)
    for key in self._properties:
        if key != self._job_type:
            default_job_properties[key] = self._properties[key]

    default_job_properties[self._job_type].update(self._properties[self._job_type])
    new_job_config._properties = default_job_properties

    return new_job_config


def _handle_request(self, request: dict) -> dict:

    request_body: bytes = request["request_body"]
    signature_chain_url: str = request["signature_chain_url"]
    signature: str = request["signature"]
    alexa_request: dict = request["alexa_request"]

    if not self._verify_request(signature_chain_url, signature, request_body):
        return {"error": "failed certificate/signature check"}

    timestamp_str = alexa_request["request"]["timestamp"]
    timestamp_datetime = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%SZ")
    now = datetime.utcnow()

    delta = now - timestamp_datetime if now >= timestamp_datetime else timestamp_datetime - now

    if abs(delta.seconds) > REQUEST_TIMESTAMP_TOLERANCE_SECS:
        log.error(f'Failed timestamp check for request: {request_body.decode("utf-8", "replace")}')
        return {"error": "failed request timestamp check"}

    conversation_key = alexa_request["session"]["user"]["userId"]

    if conversation_key not in self.conversations.keys():
        if self.config["multi_instance"]:
            conv_agent = self._init_agent()
            log.info("New conversation instance level agent initiated")
        else:
            conv_agent = self.agent

        self.conversations[conversation_key] = Conversation(
            config=self.config,
            agent=conv_agent,
            conversation_key=conversation_key,
            self_destruct_callback=lambda: self._del_conversation(conversation_key),
        )

        log.info(f"Created new conversation, key: {conversation_key}")

    conversation = self.conversations[conversation_key]
    response = conversation.handle_request(alexa_request)

    return response


def proba2onehot(proba: [list, np.ndarray], confident_threshold: float, classes: [list, np.ndarray]) -> np.ndarray:

    return labels2onehot(proba2labels(proba, confident_threshold, classes), classes)


def check_partition_column(partition_column, cols):

    for k, v in cols.items():
        if k == partition_column:
            if v == "int":
                return
            else:
                raise InvalidPartitionColumn("partition_column must be int, and not {0}".format(v))
    raise InvalidPartitionColumn("partition_column {0} not found in the query".format(partition_column))


def __init__(self, name=None):

    if not name or name[-1] != "/":  # `name` is not a name scope.
        with tf.compat.v1.name_scope(name or type(self).__name__) as name:
            pass
    self._name = name
