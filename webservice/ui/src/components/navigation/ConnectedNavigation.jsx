import { connect } from "react-redux";
import Navigation from "./Navigation";
import { toggleStats,toggleStreaming } from "../../dux/stats";

// maps the redux state to this components props
const mapStateToProps = state => ( {
  statsOn: state.stats.statsOn,
} );

// provide the component with the dispatch method
const mapDispatchToProps = dispatch => ( {
  toggleStats: () => {
    dispatch( toggleStats() );
  },
  toggleStreaming:()=>{
    dispatch(toggleStreaming());
  }
} );

export default connect( mapStateToProps, mapDispatchToProps )( Navigation );
