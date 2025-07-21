def _init_and_record(
    self,
    channel,
    record_from,
    ref_channel,
    mechanism=None,
):
    section_names = list(self.sections.keys())

    # E.g., with channel="ik" and record_from="all" ...
    # we are setting the attribute to be self.ik = dict.fromkeys(section_names)
    if record_from == "soma":
        setattr(
            self,
            channel,
            dict.fromkeys(["soma"]),
        )
    elif record_from == "all":
        setattr(
            self,
            channel,
            dict.fromkeys(section_names),
        )
    else:
        return

    # E.g., with channel="ik" and record_from="all" ...
    # we are getting the self.ik initialized above
    record_dict = getattr(
        self,
        channel,
        None,
    )
    if record_dict is None:
        return

    # E.g., with channel="ik" and record_from="all" ...
    # Looping through the keys (section names) of self.ik
    for sec_name in record_dict:
        # Get the middle segment of the section
        segment = self._nrn_sections[sec_name](0.5)
        # if the channel recording is not exposed at the segment level,
        # reference the appropriate .mod file mechanism
        #
        # e.g., aggregate potassium ("ik") currents are exposed
        # at the segment level. the individual "ik" channel currents
        # are exposed by their respective mod files, and those
        # mechanisms (e.g., "hh2", "kca", "km") are attributes of
        # the segment
        if mechanism:
            # check that segment has the mechanism and channel, which is
            # exposed with the "_ref" prefix in NEURON
            if hasattr(segment, mechanism) and hasattr(
                getattr(segment, mechanism), ref_channel
            ):
                vec = h.Vector()
                # E.g., with channel="ik", record_from="all", mechanism="hh2" ...
                # vec.record()
                vec.record(getattr(getattr(segment, mechanism), ref_channel))
                record_dict[sec_name] = vec
        else:
            if hasattr(segment, ref_channel):
                vec = h.Vector()
                vec.record(getattr(segment, ref_channel))
                record_dict[sec_name] = vec
