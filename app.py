# import keras.backend.tensorflow_backend as tb
import streamlit as st
from st_functions import section_zero, pars_arg, introduction, \
    section_one, section_two, section_three, section_four, \
    section_five


def main():
    #tb._SYMBOLIC_SCOPE.value = True

    section_zero()
    args = pars_arg()

    introduction_button = st.sidebar.checkbox('1) Introduction', key=None)

    if introduction_button:
        introduction()

    load_select = st.sidebar.checkbox('2) Load Imageset', key=None)

    if load_select:
        section_one(args)

    vis_select = st.sidebar.checkbox('3) Visualize Imageset', key=None)

    if vis_select:
        section_two()

    cluster_select = st.sidebar.checkbox('4) Cluster Imageset', key=None)

    if cluster_select:
        section_three()

    vis_cluster_select = st.sidebar.checkbox('5) Vis. Clusters & Label Imageset', key=None)

    if vis_cluster_select:
        section_four()

    performance_select = st.sidebar.checkbox('6) Clustering Performance Analytics', key=None)

    if performance_select:
        section_five()

    st.sidebar.markdown('**_For optimum performance keep '
                        'only one section active/visible at a time_**')


if __name__ == '__main__':
    main()
