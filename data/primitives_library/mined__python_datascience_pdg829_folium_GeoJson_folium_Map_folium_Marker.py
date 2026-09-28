# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg829::folium.GeoJson+folium.Map+folium.Marker
# name: folium_geojson_primitive
# summary: Uses folium.GeoJson, folium.Map, folium.Marker, folium.Popup across 2 repos
# anchor_symbols: ['folium.GeoJson', 'folium.Map', 'folium.Marker', 'folium.Popup', 'geojson.LineString']
# observed in 2 repos: ['bonzanini__Book-SocialMediaMiningPython', 'scikit-mobility__scikit-mobility']...

# --- from bonzanini__Book-SocialMediaMiningPython::Chap02-03/twitter_map_basic.py::make_map ---
def make_map(geojson_file, map_file):
    tweet_map = folium.Map(location=[50, 5],
                           zoom_start=5)
    geojson_layer = folium.GeoJson(open(geojson_file),
                                   name='geojson')
    geojson_layer.add_to(tweet_map)
    tweet_map.save(map_file)

# --- from bonzanini__Book-SocialMediaMiningPython::Chap02-03/twitter_map_clustered.py::make_map ---
def make_map(geojson_file, map_file):
    tweet_map = folium.Map(location=[50, 5],
                           zoom_start=5)
    marker_cluster = folium.MarkerCluster().add_to(tweet_map)

    geojson_layer = folium.GeoJson(open(geojson_file),
                                   name='geojson')
    geojson_layer.add_to(marker_cluster)
    tweet_map.save(map_file)

# --- from scikit-mobility__scikit-mobility::skmob/utils/plot.py::plot_flows ---
def plot_flows(fdf, map_f=None, min_flow=0, tiles='cartodbpositron', zoom=6, flow_color='red', opacity=0.5,
               flow_weight=5, flow_exp=0.5, style_function=flow_style_function,
               flow_popup=False, num_od_popup=5, tile_popup=True, radius_origin_point=5,
               color_origin_point='#3186cc', control_scale=True):
    """
    :param fdf: FlowDataFrame
        `FlowDataFrame` to visualize.

    :param map_f: folium.Map
        `folium.Map` object where the flows will be plotted. If `None`, a new map will be created.

    :param min_flow: float
        only flows larger than `min_flow` will be plotted.

    :param tiles: str
        folium's `tiles` parameter.

    :param zoom: int
        initial zoom.

    :param flow_color: str
        color of the flow edges

    :param opacity: float
        opacity (alpha level) of the flow edges.

    :param flow_weight: float
        weight factor used in the function to compute the thickness of the flow edges.

    :param flow_exp: float
        weight exponent used in the function to compute the thickness of the flow edges.

    :param style_function: lambda function
        GeoJson style function.

    :param flow_popup: bool
        if `True`, when clicking on a flow edge a popup window displaying information on the flow will appear.

    :param num_od_popup: int
        number of origin-destination pairs to show in the popup window of each origin location.

    :param tile_popup: bool
        if `True`, when clicking on a location marker a popup window displaying information on the flows
        departing from that location will appear.

    :param radius_origin_point: float
        size of the location markers.

    :param color_origin_point: str
        color of the location markers.

    :param control_scale: bool
        if `True`, add scale information in the bottom left corner of the visualization. The default is `True`.

    Returns
    -------
        `folium.Map` object with the plotted flows.

    """
    if map_f is None:
        # initialise map
        lon, lat = np.mean(np.array(list(fdf.tessellation.geometry.apply(utils.get_geom_centroid).values)), axis=0)
        map_f = folium.Map(location=[lat,lon], tiles=tiles, zoom_start=zoom, control_scale=control_scale)

    mean_flows = fdf[constants.FLOW].mean()

    O_groups = fdf.groupby(by=constants.ORIGIN)
    for O, OD in O_groups:

        geom = fdf.get_geometry(O)
        lonO, latO = utils.get_geom_centroid(geom)

        for D, T in OD[[constants.DESTINATION, constants.FLOW]].values:
            if O == D:
                continue
            if T < min_flow:
                continue

            geom = fdf.get_geometry(D)
            lonD, latD = utils.get_geom_centroid(geom)

            gjc = LineString([(lonO,latO), (lonD,latD)])

            fgeojson = folium.GeoJson(gjc,
                                      name='geojson',
                                      style_function = style_function(T / mean_flows, flow_color, opacity,
                                                                      flow_weight, flow_exp)
                                      )
            if flow_popup:
                popup = folium.Popup('flow from %s to %s: %s'%(O, D, int(T)), max_width=300)
                fgeojson = fgeojson.add_child(popup)

            fgeojson.add_to(map_f)

    if radius_origin_point > 0:
        for O, OD in O_groups:

            name = 'origin: %s' % O.replace('\'', '_')
            T_D = [[T, D] for D, T in OD[[constants.DESTINATION, constants.FLOW]].values]
            trips_info = '<br/>'.join(["flow to %s: %s" %
                                       (dd.replace('\'', '_'), int(tt)) \
                                       for tt, dd in sorted(T_D, reverse=True)[:num_od_popup]])

            geom = fdf.get_geometry(O)
            lonO, latO = utils.get_geom_centroid(geom)
            fmarker = folium.CircleMarker([latO, lonO],
                                          radius=radius_origin_point,
                                          weight=2,
                                          color=color_origin_point,
                                          fill=True, fill_color=color_origin_point
                                          )
            if tile_popup:
                popup = folium.Popup(name+'<br/>'+trips_info, max_width=300)
                fmarker = fmarker.add_child(popup)
            fmarker.add_to(map_f)

    return map_f

# --- from scikit-mobility__scikit-mobility::skmob/utils/plot.py::plot_trajectory ---
def plot_trajectory(tdf, map_f=None, max_users=None, max_points=1000, style_function=traj_style_function,
                    tiles='cartodbpositron', zoom=12, hex_color=None, weight=2, opacity=0.75, dashArray='0, 0',
                    start_end_markers=True, control_scale=True):


    """
    :param tdf: TrajDataFrame
         TrajDataFrame to be plotted.

    :param map_f: folium.Map
        `folium.Map` object where the trajectory will be plotted. If `None`, a new map will be created.

    :param max_users: int
        maximum number of users whose trajectories should be plotted.

    :param max_points: int
        maximum number of points per user to plot.
        If necessary, a user's trajectory will be down-sampled to have at most `max_points` points.

    :param style_function: lambda function
        function specifying the style (weight, color, opacity) of the GeoJson object.

    :param tiles: str
        folium's `tiles` parameter.

    :param zoom: int
        initial zoom.

    :param hex_color: str
        hex color of the trajectory line. If `None` a random color will be generated for each trajectory.

    :param weight: float
        thickness of the trajectory line.

    :param opacity: float
        opacity (alpha level) of the trajectory line.

    :param dashArray: str
        style of the trajectory line: '0, 0' for a solid trajectory line, '5, 5' for a dashed line
        (where dashArray='size of segment, size of spacing').

    :param start_end_markers: bool
        add markers on the start and end points of the trajectory.

    :param control_scale: bool
        if `True`, add scale information in the bottom left corner of the visualization. The default is `True`.

    Returns
    -------
        `folium.Map` object with the plotted trajectories.

    """
    if max_users is None:
        max_users = 10
        warnings.warn("Only the trajectories of the first 10 users will be plotted. Use the argument `max_users` to specify the desired number of users, or filter the TrajDataFrame.", stacklevel=STACKLEVEL)

    # group by user and keep only the first `max_users`
    nu = 0

    try:
        # column 'uid' is present in the TrajDataFrame
        groups = tdf.groupby(constants.UID)
    except KeyError:
        # column 'uid' is not present
        groups = [[None, tdf]]

    warned = False
    for user, df in groups:

        if nu >= max_users:
            break
        nu += 1

        traj = df[[constants.LONGITUDE, constants.LATITUDE]]

        if max_points is None:
            di = 1
        else:
            if not warned: 
                warnings.warn("If necessary, trajectories will be down-sampled to have at most `max_points` points. To avoid this, specify `max_points=None`.", stacklevel=STACKLEVEL)
                warned = True
            di = max(1, len(traj) // max_points)
        traj = traj[::di]

        if nu == 1 and map_f is None:
            # initialise map
            center = list(np.median(traj, axis=0)[::-1])
            map_f = folium.Map(location=center, zoom_start=zoom, tiles=tiles, control_scale=control_scale)

        trajlist = traj.values.tolist()
        line = LineString(trajlist)

        if hex_color is None:
            color = get_color(-2)
        else:
            color = hex_color

        tgeojson = folium.GeoJson(line,
                                  name='tgeojson',
                                  style_function=style_function(weight, color, opacity, dashArray)
                                  )
        tgeojson.add_to(map_f)

        if start_end_markers:

            dtime, la, lo = df.loc[df['datetime'].idxmin()]\
                [[constants.DATETIME, constants.LATITUDE, constants.LONGITUDE]].values
            dtime = dtime.strftime('%Y/%m/%d %H:%M')
            mker = folium.Marker(trajlist[0][::-1], icon=folium.Icon(color='green'))
            popup = folium.Popup('<i>Start</i><BR>{}<BR>Coord: <a href="https://www.google.co.uk/maps/place/{},{}" target="_blank">{}, {}</a>'.\
                          format(dtime, la, lo, np.round(la, 4), np.round(lo, 4)), max_width=300)
            mker = mker.add_child(popup)
            mker.add_to(map_f)

            dtime, la, lo = df.loc[df['datetime'].idxmax()]\
                [[constants.DATETIME, constants.LATITUDE, constants.LONGITUDE]].values
            dtime = dtime.strftime('%Y/%m/%d %H:%M')
            mker = folium.Marker(trajlist[-1][::-1], icon=folium.Icon(color='red'))
            popup = folium.Popup('<i>End</i><BR>{}<BR>Coord: <a href="https://www.google.co.uk/maps/place/{},{}" target="_blank">{}, {}</a>'.\
                          format(dtime, la, lo, np.round(la, 4), np.round(lo, 4)), max_width=300)
            mker = mker.add_child(popup)
            mker.add_to(map_f)

    return map_f
