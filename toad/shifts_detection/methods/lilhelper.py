class finfo:
    """
    finfo(dtype)

    Machine limits for floating point types.
    """

    def _init(self, dtype):
        #self.dtype = numeric.dtype(dtype)
        machar = _get_machar(dtype)

        #for word in ['precision', 'iexp',
        #             'maxexp', 'minexp', 'negep',
        #             'machep']:
        #    setattr(self, word, getattr(machar, word))
        #for word in ['resolution', 'epsneg', 'smallest_subnormal']:
        #    setattr(self, word, getattr(machar, word).flat[0])
        #self.bits = self.dtype.itemsize * 8
        #self.max = machar.huge.flat[0]
        #self.min = -self.max
        self.eps = machar.eps.flat[0]
        #self.nexp = machar.iexp
        #self.nmant = machar.it
        #self._machar = machar
        #self._str_tiny = machar._str_xmin.strip()
        #self._str_max = machar._str_xmax.strip()
        #self._str_epsneg = machar._str_epsneg.strip()
        #self._str_eps = machar._str_eps.strip()
        #self._str_resolution = machar._str_resolution.strip()
        #self._str_smallest_normal = machar._str_smallest_normal.strip()
        #self._str_smallest_subnormal = machar._str_smallest_subnormal.strip()
        return self
    
def _get_machar(ftype):
    """ Get MachAr instance or MachAr-like instance

    Get parameters for floating point type, by first trying signatures of
    various known floating point types, then, if none match, attempting to
    identify parameters by analysis.

    """
    _KNOWN_TYPES = {}
    _MACHAR_PARAMS = {
        ntypes.double: {
            'itype': ntypes.int64,
            'fmt': '%24.16e',
            'title': _title_fmt.format('double')},
        ntypes.single: {
            'itype': ntypes.int32,
            'fmt': '%15.7e',
            'title': _title_fmt.format('single')},
        ntypes.longdouble: {
            'itype': ntypes.longlong,
            'fmt': '%s',
            'title': _title_fmt.format('long double')},
        ntypes.half: {
            'itype': ntypes.int16,
            'fmt': '%12.5e',
            'title': _title_fmt.format('half')}}

    params = _MACHAR_PARAMS.get(ftype)
    if params is None:
        raise ValueError(repr(ftype))
    # Detect known / suspected types
    # ftype(-1.0) / ftype(10.0) is better than ftype('-0.1') because stold
    # may be deficient
    key = (ftype(-1.0) / ftype(10.))
    key = key.view(key.dtype.newbyteorder("<")).tobytes()
    ma_like = None
    if ftype == ntypes.longdouble:
        # Could be 80 bit == 10 byte extended precision, where last bytes can
        # be random garbage.
        # Comparing first 10 bytes to pattern first to avoid branching on the
        # random garbage.
        ma_like = _KNOWN_TYPES.get(key[:10])
    if ma_like is None:
        # see if the full key is known.
        ma_like = _KNOWN_TYPES.get(key)
    if ma_like is None and len(key) == 16:
        # machine limits could be f80 masquerading as np.float128,
        # find all keys with length 16 and make new dict, but make the keys
        # only 10 bytes long, the last bytes can be random garbage
        _kt = {k[:10]: v for k, v in _KNOWN_TYPES.items() if len(k) == 16}
        ma_like = _kt.get(key[:10])
    if ma_like is not None:
        return ma_like
    # Fall back to parameter discovery
    #warnings.warn(
    #    f'Signature {key} for {ftype} does not match any known type: '
    #    'falling back to type probe function.\n'
    #    'This warnings indicates broken support for the dtype!',
    #    UserWarning, stacklevel=2)
    return _discovered_machar(ftype)