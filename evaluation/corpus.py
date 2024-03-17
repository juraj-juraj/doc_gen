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

    Raises:
        InvalidDeckMetainfo: Raise when name in protobuf is blank
        InvalidDeckVersion: Raises if version in protobuf and in argument don't match

    Returns:
        dict: Metainfo containing 'version', 'name', 'issue_mode', 'number_of_decimals' and 'asset_specific_data'
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
