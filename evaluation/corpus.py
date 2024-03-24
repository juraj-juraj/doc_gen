def _get_from_registry(self):
    """Get hkey local hkey from windows registries
    Lookup each location and look for current version of java runtime environment runtime lib
    Winreg is an optional package in cygwin

    Returns:
        Version of installed java runtime environment or None



    Lookup version of runtime lib of installed java runtime environment

    Returns:
        Jre version installed or None
    """
    winreg = self._get_winreg()
    if not winreg:
        return None
    for location in reg_keys:
        try:
            jreKey = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, location)
            cv = winreg.QueryValueEx(jreKey, "CurrentVersion")
            versionKey = winreg.OpenKey(jreKey, cv[0])
            winreg.CloseKey(jreKey)

            cv = winreg.QueryValueEx(versionKey, "RuntimeLib")
            winreg.CloseKey(versionKey)

            return cv[0]
        except WindowsError:
            pass


def parse_deckspawn_metainfo(protobuf: bytes, version: int) -> dict:
    """Parse proto message
    Parser protobuf message supplied and checks if version and name are valid

    Args:
        protobuf (bytes): Message to parse and check
        version (int): Supposed version of protobuf message

    Raises:
        InvalidDeckMetainfo: Raises in case of blank 'name' in protobuf
        InvalidDeckVersion: Raises in case of version mismatch between protobuf and supplied version

    Returns:
        dict: Metadata of protobuf



    Get metainfo from protobuf
    Parse protobuf as a 'DeckSpawnProto' message, check validity and return metainfo

    Args:
        protobuf (bytes): Bytes of supposed message
        version (int): Version to check protobuf message against

    Returns:
        dict: Metainfo containing 'version', 'name', 'issue_mode', 'number_of_decimals' and 'asset_specific_data'

    Raises:
        InvalidDeckMetainfo: Raise when name in protobuf is blank
        InvalidDeckVersion: Raises if version in protobuf and in argument don't match
    """

    deck = DeckSpawnProto()
    deck.ParseFromString(protobuf)

    error = {"error": "Deck ({deck}) metainfo incomplete, deck must have a name.".format(deck=deck.name)}

    if deck.name == "":
        raise InvalidDeckMetainfo(error)

    if deck.version != version:
        raise InvalidDeckVersion({"error", "Deck version mismatch."})

    return {
        "version": deck.version,
        "name": deck.name,
        "issue_mode": deck.issue_mode,
        "number_of_decimals": deck.number_of_decimals,
        "asset_specific_data": deck.asset_specific_data,
    }


def normalize_username(text, encoding="utf-8"):
    """Encode users name to URL compliant symbols
    Encode special characters in users name so the can be used in URL path

    Args:
        text (str): User name with special characters
        encoding (str, optional): Encoding of user name. Defaults to "utf-8".

    Returns:
        str: Encoded URL, where special characters are encoded with percent symbol



    Encode "uppercase's" in text to URL-encoded format

    Args:
        text (str): Text in specified encoding
        encoding (str, optional): Encoding of supplied text. Defaults to "utf-8".

    Returns:
        str: Encoded text as a URL path
    """

    path = percent_encode(text, encoding=encoding, encode_set=USERNAME_ENCODE_SET)
    return uppercase_percent_encoding(path)


def ssl_wrap_socket(
    socket: socket.socket,
    ssl_options: Union[Dict[str, Any], ssl.SSLContext],
    server_hostname: str = None,
    **kwargs: Any,
) -> ssl.SSLSocket:
    """Creates socket from ssl_options and uses it to wrap submitted socket
    May you want to add extra arguments to creating socket wrapper, you can using kwargs

    Args:
        socket (socket.socket): Original socket, which will be wrapped
        ssl_options (Union[Dict[str, Any], ssl.SSLContext]): Options of wrapping socket
        server_hostname (str, optional): Server hostname of new wrapper socket. Defaults to None.

    Returns:
        ssl.SSLSocket: Socket wrapping original socket from arguments



    Creates socket from ssl_options and uses it to wrap submitted socket
    May you want to add extra arguments to creating socket wrapper, you can using kwargs

    Args:
        socket (socket.socket): Original socket, which will be wrapped
        ssl_options (Union[Dict[str, Any], ssl.SSLContext]): Options of wrapping socket
        server_hostname (str, optional): Server hostname of new wrapper socket. Defaults to None.

    Returns:
        ssl.SSLSocket: Socket wrapping original socket from arguments
    """
    context = ssl_options_to_context(ssl_options)
    if ssl.HAS_SNI:
        # In python 3.4, wrap_socket only accepts the server_hostname
        # argument if HAS_SNI is true.
        # TODO: add a unittest (python added server-side SNI support in 3.4)
        # In the meantime it can be manually tested with
        # python3 -m tornado.httpclient https://sni.velox.ch
        return context.wrap_socket(socket, server_hostname=server_hostname, **kwargs)
    else:
        return context.wrap_socket(socket, **kwargs)


def _get_cn_cert():
    """Finds certificate
    Certificate should be used for jwt validation
    If certificate is not cached, it is downloaded and stored in cache

    Returns:
        certificate object, which can be used for validation



    Find certificate in django's cache
    If certificate is not present in cache, it shall be downloaded and saved to cache

    Returns:
        certificate for jwt validation
    """

    try:
        cert_obj = django.core.cache.cache.cn_cert_obj
        d1_common.cert.x509.log_cert_info(logging.debug, "Using cached CN cert for JWT validation", cert_obj)
        return cert_obj
    except AttributeError:
        cn_cert_obj = _download_and_decode_cn_cert()
        django.core.cache.cache.cn_cert_obj = cn_cert_obj
        return cn_cert_obj


def _exponential_timeout_generator(initial, maximum, multiplier, deadline):
    """A generator that yields exponential timeout values.

    Args:
        initial (float): The initial timeout.
        maximum (float): The maximum timeout.
        multiplier (float): The multiplier applied to the timeout.
        deadline (float): The overall deadline across all invocations.

    Yields:
        float: A timeout value.



    Generator yields exponential timeout values.
    It calculates timeout based on invocations, maximum and deadline.
    If deadline is not set, it is set to maximum value.

    Args:
        initial (float): The initial timeout.
        maximum (float): The maximum timeout.
        multiplier (float): The multiplier applied to the timeout.
        deadline (float): The overall deadline across all invocations.
    Yield:
        float: A timeout value in seconds.
    """
    if deadline is not None:
        deadline_datetime = datetime_helpers.utcnow() + datetime.timedelta(seconds=deadline)
    else:
        deadline_datetime = datetime.datetime.max

    timeout = initial
    while True:
        now = datetime_helpers.utcnow()
        yield min(
            # The calculated timeout based on invocations.
            timeout,
            # The set maximum timeout.
            maximum,
            # The remaining time before the deadline is reached.
            float((deadline_datetime - now).seconds),
        )
        timeout = timeout * multiplier


def verify_sans(amazon_cert: crypto.X509) -> bool:
    """Verifies Subject Alternative Names (SANs) for Amazon certificate.

    Args:
        amazon_cert (crypto.X509): Pycrypto X509 Amazon certificate.

    Returns:
        result: True if verification was successful, False if not.



    Verify san for Amazon certificate.
    It checks if san contains echo-api.amazon.com

    Args:
        amazon_cert (crypto.X509): Crypto X509 certificate.
    Returns:
        bool : Result of verification process.
    """
    cert_extentions = [amazon_cert.get_extension(i) for i in range(amazon_cert.get_extension_count())]
    subject_alt_names = ""

    for extention in cert_extentions:
        if "subjectAltName" in str(extention.get_short_name()):
            subject_alt_names = extention.__str__()
            break

    result = "echo-api.amazon.com" in subject_alt_names

    return result


def from_rfc3339_nanos(value):
    """Convert a nanosecond-precision timestamp to a native datetime.

    Please note that Python datetimes do not support nanosecond precision; this function
    therefore truncates such values to microseconds.

    Args:
        value (str): The RFC3339 string to convert.

    Returns:
        datetime.datetime: The datetime object equivalent to the timestamp in
            UTC.

    Raises:
        ValueError: If the timestamp does not match the RFC 3339
            regular expression.



    Convert rfc3339 timestamp to datetime object.
    Nanosecond precision is not supported by Python datetimes, so it is truncated to microseconds.

    Args:
        value (str): The RFC3339 string to convert.

    Returns:
        datetime.datetime: datetime converted from timestamp in UTC.

    Raises:
        ValueError: If the timestamp is corrupted and does not match the RFC 3339 specification.
    """
    with_nanos = _RFC3339_NANOS.match(value)

    if with_nanos is None:
        raise ValueError("Timestamp: {!r}, does not match pattern: {!r}".format(value, _RFC3339_NANOS.pattern))

    bare_seconds = datetime.datetime.strptime(with_nanos.group("no_fraction"), _RFC3339_NO_FRACTION)
    fraction = with_nanos.group("nanos")

    if fraction is None:
        micros = 0
    else:
        scale = 9 - len(fraction)
        nanos = int(fraction) * (10**scale)
        micros = nanos // 1000

    return bare_seconds.replace(microsecond=micros, tzinfo=pytz.utc)


def PrintMessage(self, message):
    """Convert protobuf message to text format.

    Args:
    message: The protocol buffers message.



    Parse and print fields of protobuf message.
    If use_index_order is set, fields are sorted by index.

    Args:
        message: Message with fields to print.
    """
    if (
        message.DESCRIPTOR.full_name == _ANY_FULL_TYPE_NAME
        and self.descriptor_pool
        and self._TryPrintAsAnyMessage(message)
    ):
        return
    fields = message.ListFields()
    if self.use_index_order:
        fields.sort(key=lambda x: x[0].index)
    for field, value in fields:
        if _IsMapEntry(field):
            for key in sorted(value):
                # This is slow for maps with submessage entires because it copies the
                # entire tree.  Unfortunately this would take significant refactoring
                # of this file to work around.
                entry_submsg = value.GetEntryClass()(key=key, value=value[key])
                self.PrintField(field, entry_submsg)
        elif field.label == descriptor.FieldDescriptor.LABEL_REPEATED:
            for element in value:
                self.PrintField(field, element)
        else:
            self.PrintField(field, value)


def tensor_info_proto_maps_match(map_a, map_b):
    """Whether two signature inputs/outputs match in dtype, shape and sparsity.

    Args:
        map_a: A proto map<string,TensorInfo>.
        map_b: A proto map<string,TensorInfo>.

    Returns:
        A boolean whether `map_a` and `map_b` tensors have the same dtype, shape and
        sparsity.



    Check if two signature inputs/outputs match in data type, shape and sparsity.

    Args:
        map_a: Entry to compare <string,TensorInfo>.
        map_b: Entry to compare map<string,TensorInfo>.

    Returns:
        bool: Whether `map_a` and `map_b` tensors have the same schema.
    """
    iter_a = sorted(parse_tensor_info_map(map_a).items())
    iter_b = sorted(parse_tensor_info_map(map_b).items())
    if len(iter_a) != len(iter_b):
        return False  # Mismatch count.
    for info_a, info_b in zip(iter_a, iter_b):
        if info_a[0] != info_b[0]:
            return False  # Mismatch keys.
        if _is_sparse(info_a[1]) != _is_sparse(info_b[1]):
            return False
        if info_a[1].dtype != info_b[1].dtype:
            return False
        if not _shape_match(info_a[1].get_shape(), info_b[1].get_shape()):
            return False
    return True


def find_by_id(self, submission_id):
    """Finds submission by ID.

    Args:
        submission_id: ID of the submission

    Returns:
        SubmissionDescriptor with information about submission or None if
        submission is not found.



    Find attack by its submission id.
    Attack can be found in attacks, defenses or targeted_attacks.

    Args:
        submission_id: ID of the attack to find

    Returns:
        Attack with given submission id or None if not found.
    """
    return self._attacks.get(
        submission_id, self._defenses.get(submission_id, self._targeted_attacks.get(submission_id, None))
    )


def __init__(self, temp_dir, use_gpu):
    """Initializes instance of SubmissionValidator.

    Args:
        temp_dir: temporary working directory
        use_gpu: whether to use GPU



    Initialize instance of SubmissionValidator.
    Input, output, submission and extracted directories are created in temp_dir.

    Args:
        temp_dir (str): Store files in this directory
        use_gpu (bool): Use Gpu for processing
    """
    self._temp_dir = temp_dir
    self._use_gpu = use_gpu
    self._tmp_extracted_dir = os.path.join(self._temp_dir, "tmp_extracted")
    self._extracted_submission_dir = os.path.join(self._temp_dir, "extracted")
    self._sample_input_dir = os.path.join(self._temp_dir, "input")
    self._sample_output_dir = os.path.join(self._temp_dir, "output")


def decode_base64_dict(data):
    """Decode a base64 encoded array into a NumPy array.

    Args:
        data (dict) : encoded array data to decode

    Data should have the format encoded by :func:`encode_base64_dict`.

    Returns:
        np.ndarray



    Decode encoded array in data into NumPy array.
    Data has to have entry for __ndarray__, which holds encoded numpy array.
    This function also reshapes array if it has more than one dimension.
    The shape of the array is stored in data under key 'shape'.
    Datatype of array is stored in data under key 'dtype', numbers in array will be converted upon creating.

    Args:
        data (dict) : dictionary with encoded array data

    Returns:
        numpy.ndarray : decoded array
    """
    b64 = base64.b64decode(data["__ndarray__"])
    array = np.copy(np.frombuffer(b64, dtype=data["dtype"]))
    if len(data["shape"]) > 1:
        array = array.reshape(data["shape"])
    return array


def _apply_gradients_and_copy(self, opt, raw_grad_list, ps_var_grads):
    """Apply averaged gradients to ps vars, and then copy the updated
    variables back to each tower.

    Args:
        opt: A tf.train.Optimizer
        raw_grad_list: Ngpu x Nvar x 2 gradient list from all towers
        ps_var_grads: Nvar x 2 (grad, ps_var)

    Returns:
        list : list of copy ops



    Applies gradients and copies updated values to variables.
    Performs gradient application and updates variables based on the provided gradients. The updated values
    are then copied to the corresponding variables across multiple towers or devices.

    Args:
        opt (tf.train.Optimizer): The optimizer to use for applying gradients.
        raw_grad_list (list): The list of raw gradients for each tower.
        ps_var_grads (list): The list of variable gradients for parameter server (PS).

    Returns:
        list: A list of variable update operations.

    Raises:
        None

    """
    # TODO do this for variables together?
    with tf.name_scope("apply_gradients"):
        var_update_ops = []
        for vid, (g, v) in enumerate(ps_var_grads):
            # TODO do we put momentum variables into local or global?
            apply_gradient_op = opt.apply_gradients([(g, v)])
            barrier = self._add_sync_queues_and_barrier("param_update_barrier_{}".format(vid), [apply_gradient_op])
            with tf.control_dependencies([barrier]), tf.device(self.cpu_device):
                updated_value = v.read_value()
                for towerid in range(self.nr_gpu):
                    var_update_ops.append(raw_grad_list[towerid][vid][1].assign(updated_value))
        return var_update_ops


def complain(distribution_name):
    """Issue a warning if `distribution_name` is installed.

    In a future release, this method will be updated to raise ImportError
    rather than just send a warning.

    Args:
        distribution_name (str): The name of the obsolete distribution.



    This function issues a warning if the distribution is installed.
    Finding distribution in installed packages will issue a warning.

    Args:
        distribution_name (str): The name of the distribution.
    """
    try:
        pkg_resources.get_distribution(distribution_name)
        warnings.warn(
            "The {pkg} distribution is now obsolete. "
            "Please `pip uninstall {pkg}`. "
            "In the future, this warning will become an ImportError.".format(pkg=distribution_name),
            DeprecationWarning,
        )
    except pkg_resources.DistributionNotFound:
        pass


def __init__(self, diag_bijector, validate_args=False, name="transform_diagonal"):
    """Instantiates the `TransformDiagonal` bijector.

    Args:
    diag_bijector: `Bijector` instance used to transform the diagonal.
    validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
    name: Python `str` name given to ops managed by this object.



    Initializes the `TransformDiagonal` bijector.

    Args:
        diag_bijector (Bijector): Bijector instance used for transformations.
        validate_args (bool): Check arguments for correctness.
        name (str): Name of this object, defaults to 'transform_diagonal'.
    """
    self._diag_bijector = diag_bijector
    super(TransformDiagonal, self).__init__(
        forward_min_event_ndims=2,
        inverse_min_event_ndims=2,
        validate_args=validate_args,
        dtype=diag_bijector.dtype,
        name=name,
    )


def _MakeMethodDescriptor(self, method_proto, service_name, package, scope, index):
    """Creates a method descriptor from a MethodDescriptorProto.

    Args:
      method_proto: The proto describing the method.
      service_name: The name of the containing service.
      package: Optional package name to look up for types.
      scope: Scope containing available types.
      index: Index of the method in the service.

    Returns:
      An initialized MethodDescriptor object.



    Instantiates a method descriptor.
    Name of the method is created by joining service name and method_proto name.
    Input and output types are retrieved from the scope and package.
    Options are set from method_proto.

    Args:
        method_proto: The proto describing the method.
        service_name: The name of the containing service.
        package: Optional package name to look up for types.
        scope: Scope containing available types.
        index: Index of the method in the service.

    Returns:
        New instance of MethodDescriptor.
    """
    full_name = ".".join((service_name, method_proto.name))
    input_type = self._GetTypeFromScope(package, method_proto.input_type, scope)
    output_type = self._GetTypeFromScope(package, method_proto.output_type, scope)
    return descriptor.MethodDescriptor(
        name=method_proto.name,
        full_name=full_name,
        index=index,
        containing_service=None,
        input_type=input_type,
        output_type=output_type,
        options=_OptionsOrNone(method_proto),
    )


def update_work_as_completed(self, worker_id, work_id, other_values=None, error=None):
    """Updates work piece in datastore as completed.

    Args:
    worker_id: ID of the worker which did the work
    work_id: ID of the work which was done
    other_values: dictionary with additonal values which should be saved
        with the work piece
    error: if not None then error occurred during computation of the work
        piece. In such case work will be marked as completed with error.

    Returns:
    whether work was successfully updated



    Marks work as completed in datastore.
    If work was claimed by different worker, it will not be updated.
    If error occurred during computation, work will be marked as completed with error.

    Args:
        worker_id: Worker which did the work.
        work_id: ID of the work which was done.
        other_values: Additional values which should be saved with the work piece.
        error: Error marks work which was processed but with error.

    Returns:
        bool: Whether work was successfully updated.
    """
    client = self._datastore_client
    try:
        with client.transaction() as transaction:
            work_key = client.key(KIND_WORK_TYPE, self._work_type_entity_id, KIND_WORK, work_id)
            work_entity = client.get(work_key, transaction=transaction)
            if work_entity["claimed_worker_id"] != worker_id:
                return False
            work_entity["is_completed"] = True
            if other_values:
                work_entity.update(other_values)
            if error:
                work_entity["error"] = text_type(error)
            transaction.put(work_entity)
    except Exception:
        return False
    return True


def traverse_data(obj, use_numpy=True, buffers=None):
    """Recursively traverse an object until a flat list is found.

    If NumPy is available, the flat list is converted to a numpy array
    and passed to transform_array() to handle ``nan``, ``inf``, and
    ``-inf``.

    Otherwise, iterate through all items, converting non-JSON items

    Args:
        obj (list) : a list of values or lists
        use_numpy (bool, optional) toggle NumPy as a dependency for testing
            This argument is only useful for testing (default: True)
        buffers (list, optional) list of buffers to store the transformed arrays
    Returns:
        list : a list of values



    Recursively traverses an object and flattens it into a list.
    Math constants like nan, inf and -inf are converted to strings.
    If NumPy is available, the flat list is converted to a numpy array.

    Args:
        obj (list): A list of values or lists.
        use_numpy (bool, optional): Tells that obj is a numpy array, then different method is used for transforming.
        buffers (list, optional): List of buffers to store the transformed arrays, used only with use_numpy.

    Returns:
        list: A list of values.
    """
    if use_numpy and all(isinstance(el, np.ndarray) for el in obj):
        return [transform_array(el, buffers=buffers) for el in obj]
    obj_copy = []
    for item in obj:
        # Check the base/common case first for performance reasons
        # Also use type(x) is float because it's faster than isinstance
        if type(item) is float:
            if math.isnan(item):
                item = "NaN"
            elif math.isinf(item):
                if item > 0:
                    item = "Infinity"
                else:
                    item = "-Infinity"
            obj_copy.append(item)
        elif isinstance(item, (list, tuple)):  # check less common type second
            obj_copy.append(traverse_data(item))
        else:
            obj_copy.append(item)
    return obj_copy


def adjust_saturation(img, saturation_factor):
    """Adjust color saturation of an image.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.

    Returns:
        PIL Image: Saturation adjusted image.

    Raises:
        TypeError: If img is not a PIL Image.
        ValueError: If saturation_factor is not a number.



    Change color saturation of an image, by an saturation factor.
    Saturation factor of 0 will give a black and white image, 1 will give the original colors.
    Saturation greater than 1 will enhance the colors.

    Args:
        img (PIL Image): Image to be adjusted.
        saturation_factor (float):  How much to adjust the saturation.

    Returns:
        PIL Image: Saturation adjusted image.

    Raises:
        TypeError: If img is not a PIL Image.
        ValueError: If saturation_factor is not a number.
    """
    if not _is_pil_image(img):
        raise TypeError("img should be PIL Image. Got {}".format(type(img)))

    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img


def to_pandas(self, is_transposed=False):
    """Convert this object into a Pandas DataFrame from the partitions.

    Args:
        is_transposed (bool) : A flag for telling this object that the external
            representation is transposed, but not the internal.

    Returns:
        pandas.DataFrame: A DataFrame from the partitions.

    Raises:
        ValueError: If objects in partitions are not serializable to pandas DataFrame or Series.



    Convert objects in partitions to pandas DataFrame.
    All objects have to be either dataframes or series, else warning will be sent to email.

    Args:
        is_transposed (bool): A flag for telling this object that the external representation is transposed.

    Returns:
        pandas.DataFrame: A DataFrame from the partitions.
    """
    # In the case this is transposed, it is easier to just temporarily
    # transpose back then transpose after the conversion. The performance
    # is the same as if we individually transposed the blocks and
    # concatenated them, but the code is much smaller.
    if is_transposed:
        return self.transpose().to_pandas(False).T
    else:
        retrieved_objects = [[obj.to_pandas() for obj in part] for part in self.partitions]
        if all(isinstance(part, pandas.Series) for row in retrieved_objects for part in row):
            axis = 0
        elif all(isinstance(part, pandas.DataFrame) for row in retrieved_objects for part in row):
            axis = 1
        else:
            ErrorMessage.catch_bugs_and_request_email(True)
        df_rows = [
            pandas.concat([part for part in row], axis=axis)
            for row in retrieved_objects
            if not all(part.empty for part in row)
        ]
        if len(df_rows) == 0:
            return pandas.DataFrame()
        else:
            return pandas.concat(df_rows)


def __init__(self, channel):
    """Initialize the statistics.

    Args:
        channel (grpc.channel): Grpc channel, holding the connection to the client.



    Initialize the ErrorStatsService.
    Initializes the ErrorStatsService object by setting up the necessary gRPC channel and configuring.
    the required unary-unary methods for listing group stats, listing events, and deleting events.

    Args:
        channel (grpc.channel): Grpc channel, holding the connection to the client.
    """
    self.ListGroupStats = channel.unary_unary(
        "/google.devtools.clouderrorreporting.v1beta1.ErrorStatsService/ListGroupStats",
        request_serializer=google_dot_devtools_dot_clouderrorreporting__v1beta1_dot_proto_dot_error__stats__service__pb2.ListGroupStatsRequest.SerializeToString,
        response_deserializer=google_dot_devtools_dot_clouderrorreporting__v1beta1_dot_proto_dot_error__stats__service__pb2.ListGroupStatsResponse.FromString,
    )
    self.ListEvents = channel.unary_unary(
        "/google.devtools.clouderrorreporting.v1beta1.ErrorStatsService/ListEvents",
        request_serializer=google_dot_devtools_dot_clouderrorreporting__v1beta1_dot_proto_dot_error__stats__service__pb2.ListEventsRequest.SerializeToString,
        response_deserializer=google_dot_devtools_dot_clouderrorreporting__v1beta1_dot_proto_dot_error__stats__service__pb2.ListEventsResponse.FromString,
    )
    self.DeleteEvents = channel.unary_unary(
        "/google.devtools.clouderrorreporting.v1beta1.ErrorStatsService/DeleteEvents",
        request_serializer=google_dot_devtools_dot_clouderrorreporting__v1beta1_dot_proto_dot_error__stats__service__pb2.DeleteEventsRequest.SerializeToString,
        response_deserializer=google_dot_devtools_dot_clouderrorreporting__v1beta1_dot_proto_dot_error__stats__service__pb2.DeleteEventsResponse.FromString,
    )


def _read(cls, **kwargs):
    """Read csv file from local disk.
    Args:
        filepath_or_buffer: The filepath of the csv file. We only support local files for now.
        kwargs: Keyword arguments in pandas.read_csv
    Returns:
        pd.DataFrame: A DataFrame from the csv file.



    Read csv file from local disk and convert it to a DataFrame.

    Args:
        kwargs: Keyword arguments in pandas.read_csv

    Returns:
        pd.DataFrame: A DataFrame from the csv file.
    """
    pd_obj = pandas.read_csv(**kwargs)
    if isinstance(pd_obj, pandas.DataFrame):
        return cls.from_pandas(pd_obj)
    if isinstance(pd_obj, pandas.io.parsers.TextFileReader):
        # Overwriting the read method should return a ray DataFrame for calls
        # to __next__ and get_chunk
        pd_read = pd_obj.read
        pd_obj.read = lambda *args, **kwargs: cls.from_pandas(pd_read(*args, **kwargs))
    return pd_obj


def recall_at_k(y_true: List[int], y_pred: List[List[np.ndarray]], k: int):
    """Calculates recall at k ranking metric.
    It is supposed that the ranking score for the true candidate goes first in the prediction.

    Args:
        y_true (List[int]): Labels. Not used in the calculation of the metric.
        y_predicted (List[List[np.ndarray]]): Predictions. Each prediction contains ranking score of all ranking candidates for the particular data sample.
        k (int): The number of top candidates to consider for recall calculation.

    Returns:
        float : Recall at k metric value.



    Calculate metric recall at k from predictions.

    Args:
        y_true (List[int]): Labels, not used in the calculation.
        y_pred (List[List[np.ndarray]]): Predictions. Each prediction contains ranking score of all ranking candidates for the particular data sample.
        k (int): The number of top candidates to consider for recall calculation.

    Returns:
        float: Recall at k metric value.
    """
    num_examples = float(len(y_pred))
    predictions = np.array(y_pred)
    predictions = np.flip(np.argsort(predictions, -1), -1)[:, :k]
    num_correct = 0
    for el in predictions:
        if 0 in el:
            num_correct += 1
    return float(num_correct) / num_examples


def TryConsume(self, token):
    """Tries to consume a given piece of text.

    Args:
        token (str) : Text to consume.

    Returns:
        True iff the text was consumed.



    Tries if token is same as the next token in the stream.

    Args:
        token (str): Possibly next token in stream.

    Returns:
        bool: True if token was consumed.
    """
    if self.token == token:
        self.NextToken()
        return True
    return False


def softplus_inverse(x, name=None):
    """Computes the inverse softplus, i.e., x = softplus_inverse(softplus(x)).

    Mathematically this op is equivalent to:

    ```none
    softplus_inverse = log(exp(x) - 1.)
    ```

    Args:
        x (Tensor) : Non-negative (not enforced), floating-point.
        name (Optional(str)) : A name for the operation.

    Returns:
        Tensor:  Has the same type/shape as input `x`.



    Computes the inverse of the softplus function.
    This function computes the inverse of the softplus function, which maps `x` to `y` such that `y = log(expm1(x))`.
    It uses a numerically stable derivation to handle large `x` values and clamps small/large values to ensure no
    unchosen codepath results in NaN. For small `x`, it approximates `1 - exp{-x}` as `x` to avoid issues with `0`.

    Args:
        x (Tensor): A tensor containing the input values.
        name (str): Optional name for the operation.

    Returns:
        Tensor: A tensor with the same shape and data type as `x`, representing the inverse of the softplus function.
    """
    with tf.name_scope(name or "softplus_inverse"):
        x = tf.convert_to_tensor(value=x, name="x")
        # We begin by deriving a more numerically stable softplus_inverse:
        # x = softplus(y) = Log[1 + exp{y}], (which means x > 0).
        # ==> exp{x} = 1 + exp{y}                                (1)
        # ==> y = Log[exp{x} - 1]                                (2)
        #       = Log[(exp{x} - 1) / exp{x}] + Log[exp{x}]
        #       = Log[(1 - exp{-x}) / 1] + Log[exp{x}]
        #       = Log[1 - exp{-x}] + x                           (3)
        # (2) is the "obvious" inverse, but (3) is more stable than (2) for large x.
        # For small x (e.g. x = 1e-10), (3) will become -inf since 1 - exp{-x} will
        # be zero. To fix this, we use 1 - exp{-x} approx x for small x > 0.
        #
        # In addition to the numerically stable derivation above, we clamp
        # small/large values to be congruent with the logic in:
        # tensorflow/core/kernels/softplus_op.h
        #
        # Finally, we set the input to one whenever the input is too large or too
        # small. This ensures that no unchosen codepath is +/- inf. This is
        # necessary to ensure the gradient doesn't get NaNs. Recall that the
        # gradient of `where` behaves like `pred*pred_true + (1-pred)*pred_false`
        # thus an `inf` in an unselected path results in `0*inf=nan`. We are careful
        # to overwrite `x` with ones only when we will never actually use this
        # value. Note that we use ones and not zeros since `log(expm1(0.)) = -inf`.
        threshold = np.log(np.finfo(dtype_util.as_numpy_dtype(x.dtype)).eps) + 2.0
        is_too_small = tf.less(x, np.exp(threshold))
        is_too_large = tf.greater(x, -threshold)
        too_small_value = tf.math.log(x)
        too_large_value = x
        # This `where` will ultimately be a NOP because we won't select this
        # codepath whenever we used the surrogate `ones_like`.
        x = tf.where(tf.logical_or(is_too_small, is_too_large), tf.ones_like(x), x)
        y = x + tf.math.log(-tf.math.expm1(-x))  # == log(expm1(x))
        return tf.where(is_too_small, too_small_value, tf.where(is_too_large, too_large_value, y))


def map_across_blocks(self, map_func):
    """Applies `map_func` to every partition.

    Args:
        map_func (Callable) : The function to apply.

    Returns:
        A new BaseFrameManager object, the type of object that called this.



    Apply function to every partition in the object.

    Args:
        map_func (Callable): The function to apply to every partition.

    Returns:
        BaseFrameManager: A new object with the function applied to every partition.
    """
    preprocessed_map_func = self.preprocess_func(map_func)
    new_partitions = np.array(
        [[part.apply(preprocessed_map_func) for part in row_of_parts] for row_of_parts in self.partitions]
    )
    return self.__constructor__(new_partitions)


def rsub(self, other, axis="columns", level=None, fill_value=None):
    """Subtract a DataFrame/Series/scalar from this DataFrame.

    Args:
        other: The object to use to apply the subtraction to this.
        axis (str): The axis to apply the subtraction over.
        level: Mutlilevel index level to subtract over.
        fill_value: The value to fill NaNs with.

    Returns:
        A new DataFrame with the subtraciont applied.



    Subtract a DataFrame/Series/scalar from this DataFrame.

    Args:
        other: The object to subtract this from .
        axis (str): Axis along which subtraction will be computed.
        level: Mutlilevel index level to subtract over.
        fill_value: The value to fill unnormalized numbers with.

    Returns:
        DataFrame: A new DataFrame with the subtraction applied.
    """
    return self._binary_op("rsub", other, axis=axis, level=level, fill_value=fill_value)


def pbs_for_set_no_merge(document_path, document_data):
    """Make ``Write`` protobufs for ``set()`` methods.

    Args:
        document_path (str): A fully-qualified document path.
        document_data (dict): Property names and values to use for replacing a document.

    Returns:
        List: One or two ``Write`` protobuf instances for ``set()``.

    Raises:
        ValueError: If the document data contains a field deletion without specifying merge to True.



    Creates a list of Protocol Buffers (protobufs) for a set operation without merging.
    This function creates a list of protobufs for a set operation without merging. It takes a `document_path` and
    `document_data` as input. The `document_path` specifies the path of the document in the database, and the
    `document_data` is a dictionary containing the data to be set in the document.

    If the `document_data` contains any deleted fields, the function raises a `ValueError` because a delete operation
    cannot be applied in a set request without specifying a merge option. To perform a set operation with deleted
    fields, you need to specify either 'merge=True' or 'merge=[field_paths]'.

    Args:
        document_path (str): The path of the document in the database.
        document_data (dict): The data representing the document to be set.

    Returns:
        A list of protobufs representing the set operation.

    Raises:
        ValueError: If the document contains deleted fields and no merge option is specified.
    """
    extractor = DocumentExtractor(document_data)

    if extractor.deleted_fields:
        raise ValueError(
            "Cannot apply DELETE_FIELD in a set request without " "specifying 'merge=True' or 'merge=[field_paths]'."
        )

    # Conformance tests require send the 'update_pb' even if the document
    # contains only transforms.
    write_pbs = [extractor.get_update_pb(document_path)]

    if extractor.has_transforms:
        transform_pb = extractor.get_transform_pb(document_path)
        write_pbs.append(transform_pb)

    return write_pbs


def remove_unused_links(self, used):
    """Removes all saved links except the ones that are used.

    Args:
        used (list): list of used links that should not be removed.



    Removes all saved links except the ones that are used.
    This method removes all saved links, excluding the ones specified in the `used` list. It iterates over the
    saved links in the link state table and checks each link's path against the `used` list.

    Args:
        used (list): A list of used links that should not be removed.

    """
    unused = []

    self._execute("SELECT * FROM {}".format(self.LINK_STATE_TABLE))
    for row in self.cursor:
        relpath, inode, mtime = row
        inode = self._from_sqlite(inode)
        path = os.path.join(self.root_dir, relpath)

        if path in used:
            continue

        if not os.path.exists(path):
            continue

        actual_inode = get_inode(path)
        actual_mtime, _ = get_mtime_and_size(path)

        if inode == actual_inode and mtime == actual_mtime:
            logger.debug("Removing '{}' as unused link.".format(path))
            remove(path)
            unused.append(relpath)

    for relpath in unused:
        cmd = 'DELETE FROM {} WHERE path = "{}"'
        self._execute(cmd.format(self.LINK_STATE_TABLE, relpath))


def initial_value_of_masked_time_series(time_series_tensor, broadcast_mask):
    """Get the first unmasked entry of each time series in the batch.

    Args:
        time_series_tensor (tf.Tensor) : float Tensor of shape [..., num_timesteps].
        broadcast_mask (tf.tensor): bool Tensor of same shape as `time_series`.

    Returns:
        tf.Tensor: The initial value of the masked time series.

    Raises:
        NotImplementedError: If the rank of `first_unmasked_indices` is dynamic.



    Computes the initial value of a masked time series.
    This function computes the initial value of a masked time series given the `time_series_tensor` and `broadcast_mask`.
    The `time_series_tensor` is a tensor representing the time series data, and the `broadcast_mask` is a tensor
    specifying the masked entries in the time series.

    Args:
        time_series_tensor (tf.Tensor): The time series tensor.
        broadcast_mask (tf.Tensor): The broadcast mask specifying the masked entries.

    Returns:
        tf.Tensor: The initial value of the masked time series.

    Raises:
        NotImplementedError: If the rank of `first_unmasked_indices` is dynamic.
    """
    num_timesteps = tf.shape(input=time_series_tensor)[-1]

    # Compute the index of the first unmasked entry for each series in the batch.
    unmasked_negindices = tf.cast(~broadcast_mask, tf.int32) * tf.range(num_timesteps, 0, -1)
    first_unmasked_indices = num_timesteps - tf.reduce_max(input_tensor=unmasked_negindices, axis=-1)

    if first_unmasked_indices.shape.ndims is None:
        raise NotImplementedError(
            "Cannot compute initial values of a masked time series with" "dynamic rank."
        )  # `batch_gather` requires static rank

    # Extract the initial value for each series in the batch.
    return tf.squeeze(
        tf.compat.v1.batch_gather(params=time_series_tensor, indices=first_unmasked_indices[..., tf.newaxis]), axis=-1
    )


def _assert_same_base_type(items, expected_type=None):
    """Asserts all items are of the same base type.

    Args:
        items: List of graph items (e.g., `Variable`, `Tensor`, `SparseTensor`,
            `Operation`, or `IndexedSlices`). Can include `None` elements, which
            will be ignored.
        expected_type: Expected type. If not specified, assert all items are
            of the same base type.

    Returns:
        Validated type, or none if neither expected_type nor items provided.

    Raises:
        ValueError: If any types do not match.



    Check whether all items in items are the expected base type.
    Raises exception with mismatched argument name and type
    None elements will be ignored.

    Args:
        items (list): List of objects.
        expected_type: Expected type of elements in items

    Raises:
        ValueError: Exception with item that had mismatched element type
    """
    original_expected_type = expected_type
    mismatch = False
    for item in items:
        if item is not None:
            item_type = base_dtype(item.dtype)
            if not expected_type:
                expected_type = item_type
            elif expected_type != item_type:
                mismatch = True
                break
    if mismatch:
        # Loop back through and build up an informative error message (this is very
        # slow, so we don't do it unless we found an error above).
        expected_type = original_expected_type
        original_item_str = None
        get_name = lambda x: x.name if hasattr(x, "name") else str(x)
        for item in items:
            if item is not None:
                item_type = base_dtype(item.dtype)
                if not expected_type:
                    expected_type = item_type
                    original_item_str = get_name(item)
                elif expected_type != item_type:
                    raise ValueError(
                        "{}, type={}, must be of the same type ({}){}.".format(
                            get_name(item),
                            item_type,
                            expected_type,
                            ((" as {}".format(original_item_str)) if original_item_str else ""),
                        )
                    )
        return expected_type  # Should be unreachable
    else:
        return expected_type


def generate(self, model_len=None, model_width=None):
    """Generates a Multi-Layer Perceptron.
    Args:
        model_len: An integer. Number of hidden layers.
        model_width: An integer or a list of integers of length `model_len`. If it is a list, it represents the
            number of nodes in each hidden layer. If it is an integer, all hidden layers have nodes equal to this
            value.
    Returns:
        An instance of the class Graph. Represents the neural architecture graph of the generated model.



    Generates a graph-based model architecture.

    This method generates a graph-based model architecture by creating a Graph object and iteratively adding layers
    to it based on the specified parameters. The `model_len` parameter determines the number of layers in the model,
    while the `model_width` parameter specifies the width of each layer.

    Args:
        model_len (int, optional): The number of layers in the model. If not specified, the default value from
            the Constant class will be used.
        model_width (int or list, optional): The width of each layer in the model. If an int is provided, it will be
            used as the width for all layers. If a list is provided, it should have a length equal to `model_len`,
            specifying the width of each layer individually. If not specified, the default value from the Constant
            class will be used.

    Returns:
        Graph: The generated graph-based model architecture.

    Raises:
        ValueError: If the length of 'model_width' does not match 'model_len'.
    """
    if model_len is None:
        model_len = Constant.MODEL_LEN
    if model_width is None:
        model_width = Constant.MODEL_WIDTH
    if isinstance(model_width, list) and not len(model_width) == model_len:
        raise ValueError("The length of 'model_width' does not match 'model_len'")
    elif isinstance(model_width, int):
        model_width = [model_width] * model_len

    graph = Graph(self.input_shape, False)
    output_node_id = 0
    n_nodes_prev_layer = self.input_shape[0]
    for width in model_width:
        output_node_id = graph.add_layer(StubDense(n_nodes_prev_layer, width), output_node_id)
        output_node_id = graph.add_layer(StubDropout1d(Constant.MLP_DROPOUT_RATE), output_node_id)
        output_node_id = graph.add_layer(StubReLU(), output_node_id)
        n_nodes_prev_layer = width

    graph.add_layer(StubDense(n_nodes_prev_layer, self.n_output_node), output_node_id)
    return graph


def sum_rightmost_ndims_preserving_shape(x, ndims):
    """Return `Tensor` with right-most ndims summed.

    Args:
    x: the `Tensor` whose right-most `ndims` dimensions to sum
    ndims: number of right-most dimensions to sum.

    Returns:
    A `Tensor` resulting from calling `reduce_sum` on the `ndims` right-most
    dimensions. If the shape of `x` is statically known, the result will also
    have statically known shape. Otherwise, the resulting shape will only be
    known at runtime.



    Computes the sum of the rightmost dimensions of a tensor while preserving the shape.

    This function takes an input tensor `x` and computes the sum of its rightmost `ndims` dimensions while preserving
    the shape of the tensor.

    Args:
        x (tf.Tensor): The input tensor.
        ndims (int): The number of rightmost dimensions to sum.

    Returns:
        tf.Tensor: The tensor resulting from summing the rightmost dimensions of `x`.
    """
    x = tf.convert_to_tensor(value=x)
    if x.shape.ndims is not None:
        axes = tf.range(x.shape.ndims - ndims, x.shape.ndims)
    else:
        axes = tf.range(tf.rank(x) - ndims, tf.rank(x))
    return tf.reduce_sum(input_tensor=x, axis=axes)


def get_lanczos_eig(self, compute_m=True, feed_dict=None):
    """Computes the min eigen value and corresponding vector of matrix M or H
    using the Lanczos algorithm.

    Args:
        compute_m (bool): Determine whether we should compute eig val/vec
            for M or for H. True for M; False for H.
        feed_dict (dict): Mapping from TF placeholders to values (optional)
    Returns:
        min_eig_vec: Corresponding eigen vector to min eig val
        eig_val: Minimum eigen value



    Computes the minimum eigenvalue and its corresponding eigenvector using the Lanczos algorithm.

    This method utilizes the Lanczos algorithm to compute the minimum eigenvalue and its corresponding eigenvector
    for either matrix M or matrix H, based on the value of the `compute_m` parameter.

    Args:
        compute_m (bool): Determines whether to compute eigenvalue and eigenvector for matrix M or H.
            Set to True for matrix M, and False for matrix H.
        feed_dict (dict): Optional mapping from TensorFlow placeholders to values.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: A tuple containing the corresponding eigenvector (min_eig_vec) and
            the minimum eigenvalue (eig_val).
    """
    if compute_m:
        min_eig, min_vec = self.sess.run([self.m_min_eig, self.m_min_vec], feed_dict=feed_dict)

    else:
        min_eig, min_vec = self.sess.run([self.h_min_eig, self.h_min_vec], feed_dict=feed_dict)

    return min_vec, min_eig


def _AddPropertiesForField(field, cls):
    """Adds a public property for a protocol message field.
    Clients can use this property to get and (in the case
    of non-repeated scalar fields) directly set the value
    of a protocol message field.

    Args:
        field: A FieldDescriptor for this field.
        cls: The class we're constructing.



    Adds a public property for a protocol message field.

    Clients can use this property to get and, in the case of non-repeated scalar fields, directly set the value of a
    protocol message field.

    Args:
        field: FieldDescriptor for the field.
        cls: The class being constructed.
    """

    # Catch it if we add other types that we should
    # handle specially here.
    assert _FieldDescriptor.MAX_CPPTYPE == 10

    constant_name = field.name.upper() + "_FIELD_NUMBER"
    setattr(cls, constant_name, field.number)

    if field.label == _FieldDescriptor.LABEL_REPEATED:
        _AddPropertiesForRepeatedField(field, cls)
    elif field.cpp_type == _FieldDescriptor.CPPTYPE_MESSAGE:
        _AddPropertiesForNonRepeatedCompositeField(field, cls)
    else:
        _AddPropertiesForNonRepeatedScalarField(field, cls)


def moments_of_masked_time_series(time_series_tensor, broadcast_mask):
    """Compute mean and variance, accounting for a mask.

    Args:
        time_series_tensor: float `Tensor` time series of shape
        `concat([batch_shape, [num_timesteps]])`.
        broadcast_mask: bool `Tensor` of the same shape as `time_series`.
    Returns:
        mean: float `Tensor` of shape `batch_shape`.
        variance: float `Tensor` of shape `batch_shape`.



    Compute the mean and variance of a time series, accounting for a mask.

    This function computes the mean and variance of a time series tensor, taking into account a corresponding mask. The
    time series tensor has shape `concat([batch_shape, [num_timesteps]])`, where `batch_shape` represents the shape
    of the batch dimensions and `num_timesteps` represents the number of time steps.

    Args:
        time_series_tensor: A float `Tensor` representing the time series, with shape
            `concat([batch_shape, [num_timesteps]])`.
        broadcast_mask: A boolean `Tensor` of the same shape as `time_series_tensor`, representing the mask.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: A tuple containing the mean and variance as float `Tensor` objects with shape
            `batch_shape`.
    """
    num_unmasked_entries = tf.cast(
        tf.reduce_sum(input_tensor=tf.cast(~broadcast_mask, tf.int32), axis=-1), time_series_tensor.dtype
    )

    # Manually compute mean and variance, excluding masked entries.
    mean = (
        tf.reduce_sum(
            input_tensor=tf.where(broadcast_mask, tf.zeros_like(time_series_tensor), time_series_tensor), axis=-1
        )
        / num_unmasked_entries
    )
    variance = (
        tf.reduce_sum(
            input_tensor=tf.where(
                broadcast_mask, tf.zeros_like(time_series_tensor), (time_series_tensor - mean[..., tf.newaxis]) ** 2
            ),
            axis=-1,
        )
        / num_unmasked_entries
    )
    return mean, variance


def feedforward(inputs, num_units, scope="multihead_attention"):
    """Point-wise feed forward net.

    Args:
        inputs (tf.Tensor): A 3d tensor with shape of [N, T, C].
        num_units (list): A list of two integers.
        scope (str): Optional scope for `variable_scope`.

    Returns:
        tf.Tensor: A 3d tensor with the same shape and dtype as inputs



    Point-wise feed forward network.
    This function implements a point-wise feed forward network. It takes an input tensor `inputs` with shape [N, T, C],
    where N represents the batch size, T represents the number of time steps, and C represents the number of channels.

    Args:
        inputs (tf.Tensor): A 3D tensor with shape [N, T, C].
        num_units (list): A list of two integers specifying the number of units for the inner and readout layers respectively.
        scope (str): Optional scope for `variable_scope`.

    Returns:
        tf.Tensor: A 3D tensor with the same shape and dtype as the input `inputs`.
    """
    with tf.variable_scope(scope):
        # Inner layer
        params = {
            "inputs": inputs,
            "filters": num_units[0],
            "kernel_size": 1,
            "activation": tf.nn.relu,
            "use_bias": True,
        }
        outputs = tf.layers.conv1d(**params)

        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1, "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = normalize(outputs)

    return outputs


def box(self, x0, y0, width, height):
    """Create a box on ASCII canvas.

    Args:
        x0 (int): x coordinate of the box corner.
        y0 (int): y coordinate of the box corner.
        width (int): box width.
        height (int): box height.



    Generate box of given dimensions in canvas on coordinates

    Args:
        x0 (int): x coordinate of the box corner.
        y0 (int): y coordinate of the box corner.
        width (int): box width.
        height (int): box height.
    """
    assert width > 1
    assert height > 1

    width -= 1
    height -= 1

    for x in range(x0, x0 + width):
        self.point(x, y0, "-")
        self.point(x, y0 + height, "-")

    for y in range(y0, y0 + height):
        self.point(x0, y, "|")
        self.point(x0 + width, y, "|")

    self.point(x0, y0, "+")
    self.point(x0 + width, y0, "+")
    self.point(x0, y0 + height, "+")
    self.point(x0 + width, y0 + height, "+")


def main(argv=None):
    """Run dvc CLI command.

    Args:
        argv (list): optional list of arguments to parse. sys.argv is used by default.

    Returns:
        int: command's return code.



    Run the dvc CLI command.

    This function is the entry point for running the dvc CLI command. It takes an optional list of arguments, `argv`,
    which can be used to override the default arguments obtained from `sys.argv`.

    Args:
        argv (list): Optional list of arguments to parse. If not provided, sys.argv is used by default.

    Returns:
        int: The return code of the command.
    """
    args = None
    cmd = None

    try:
        args = parse_args(argv)

        if args.quiet:
            logger.setLevel(logging.CRITICAL)

        elif args.verbose:
            logger.setLevel(logging.DEBUG)

        cmd = args.func(args)
        ret = cmd.run_cmd()
    except KeyboardInterrupt:
        logger.exception("interrupted by the user")
        ret = 252
    except NotDvcRepoError:
        logger.exception("")
        ret = 253
    except DvcParserError:
        ret = 254
    except Exception:  # pylint: disable=broad-except
        logger.exception("unexpected error")
        ret = 255

    Analytics().send_cmd(cmd, args, ret)

    return ret


def __call__(self, kl_fn):
    """Perform the KL registration.

    Args:
        kl_fn (Callable): The function to use for the KL divergence.

    Returns:
        Callable: kl_fn callable object submitted as argument

    Raises:
        TypeError: if kl_fn is not a callable.
        ValueError: if a KL divergence function has already been registered for
        the given argument classes.



    Perform KL registration using the specified KL divergence function.

        This function performs KL registration by assigning the provided kl_fn as the KL divergence function for the
    specified argument classes. The kl_fn should be a callable that computes the KL divergence.

    Args:
        kl_fn (Callable): A callable representing the function to use for the KL divergence.

    Returns:
        Callable: The kl_fn callable object submitted as an argument.

    Raises:
        TypeError: If kl_fn is not a callable.
        ValueError: If a KL divergence function has already been registered for the given argument classes.
    """
    if not callable(kl_fn):
        raise TypeError("kl_fn must be callable, received: %s" % kl_fn)
    if self._key in _DIVERGENCES:
        raise ValueError(
            "KL(%s || %s) has already been registered to: %s"
            % (self._key[0].__name__, self._key[1].__name__, _DIVERGENCES[self._key])
        )
    _DIVERGENCES[self._key] = kl_fn
    return kl_fn


def build_losses(self, logits_real, logits_fake):
    """Build standard GAN loss and set `self.g_loss` and `self.d_loss`.
    D and G play two-player minimax game with value function V(G,D)
    min_G max _D V(D, G) = IE_{x ~ p_data} [log D(x)] + IE_{z ~ p_fake} [log (1 - D(G(z)))]

    Args:
        logits_real (tf.Tensor): discrim logits from real samples
        logits_fake (tf.Tensor): discrim logits from fake samples produced by generator



    Build the standard GAN loss and set `self.g_loss` and `self.d_loss`.

    The discriminator (D) and generator (G) play a two-player minimax game with a value function V(G,D). The goal is to
    minimize GAN loss (V(D, G)) defined as:
    V(D, G) = E_{x ~ p_data}[log D(x)] + E_{z ~ p_fake}[log (1 - D(G(z)))]

    Args:
        logits_real: A TensorFlow tensor representing the discriminator logits from real samples.
        logits_fake: A TensorFlow tensor representing the discriminator logits from fake samples produced by the generator.
    """
    with tf.name_scope("GAN_loss"):
        score_real = tf.sigmoid(logits_real)
        score_fake = tf.sigmoid(logits_fake)
        tf.summary.histogram("score-real", score_real)
        tf.summary.histogram("score-fake", score_fake)

        with tf.name_scope("discrim"):
            d_loss_pos = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real, labels=tf.ones_like(logits_real)),
                name="loss_real",
            )
            d_loss_neg = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.zeros_like(logits_fake)),
                name="loss_fake",
            )

            d_pos_acc = tf.reduce_mean(tf.cast(score_real > 0.5, tf.float32), name="accuracy_real")
            d_neg_acc = tf.reduce_mean(tf.cast(score_fake < 0.5, tf.float32), name="accuracy_fake")

            d_accuracy = tf.add(0.5 * d_pos_acc, 0.5 * d_neg_acc, name="accuracy")
            self.d_loss = tf.add(0.5 * d_loss_pos, 0.5 * d_loss_neg, name="loss")

        with tf.name_scope("gen"):
            self.g_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.ones_like(logits_fake)),
                name="loss",
            )
            g_accuracy = tf.reduce_mean(tf.cast(score_fake > 0.5, tf.float32), name="accuracy")

        add_moving_summary(self.g_loss, self.d_loss, d_accuracy, g_accuracy)


def linear_extrapolation_plot(log_prob_adv_array, y, file_name, min_epsilon=-10, max_epsilon=10, num_points=21):
    """Generate linear extrapolation plot.

    Args:
        log_prob_adv_array (np.array): Numpy array containing log probabilities
        y (tf.Tensor): Tf placeholder for the labels
        file_name (str): Plot filename
        min_epsilon (int): Minimum value of epsilon over the interval
        max_epsilon (int): Maximum value of epsilon over the interval
        num_points (int): Number of points used to interpolate

    Returns:
        plt.Figure: matplotlib plot of log probability array



    Generate a linear extrapolation plot.

    This function generates a linear extrapolation plot using the provided log probability array, labels, and plot
    parameters. The log probability array is a numpy array containing the log probabilities.

    Args:
        log_prob_adv_array: A numpy array containing log probabilities.
        y: A TensorFlow placeholder for the labels.
        file_name: The filename for the plot.
        min_epsilon: The minimum value of epsilon over the interval (default: -10).
        max_epsilon: The maximum value of epsilon over the interval (default: 10).
        num_points: The number of points used for interpolation (default: 21).

    Returns:
        plt.Figure: A matplotlib plot of the log probability array.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure()
    figure.canvas.set_window_title("Cleverhans: Linear Extrapolation Plot")

    correct_idx = np.argmax(y, axis=0)
    fig = plt.figure()
    plt.xlabel("Epsilon")
    plt.ylabel("Logits")
    x_axis = np.linspace(min_epsilon, max_epsilon, num_points)
    plt.xlim(min_epsilon - 1, max_epsilon + 1)
    for i in range(y.shape[0]):
        if i == correct_idx:
            ls = "-"
            linewidth = 5
        else:
            ls = "--"
            linewidth = 2
        plt.plot(x_axis, log_prob_adv_array[:, i], ls=ls, linewidth=linewidth, label="{}".format(i))
    plt.legend(loc="best", fontsize=14)
    plt.show()
    fig.savefig(file_name)
    plt.clf()
    return figure
