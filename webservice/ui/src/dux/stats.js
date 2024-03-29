
import mq from "../MqttClient";

// actions
const TOGGLE_STATS = "features/stats/TOGGLE_STATS";
const TOGGLE_COUNT = "features/stats/TOGGLE_COUNT";
const TOGGLE_STREAMING = "features/video/STREAMING";

// initial state
const initialState = {
  statsOn: false,
  totalCountOn: true,
  peopleSeen: [],
  currentCount: 0,
  currentDuration: null,
  liveStreaming:true
};
export function getPlayerState(){
 return  initialState.liveStreaming;
}
// Reducer
export default function reducer( state = initialState, action = {} ) {
  switch ( action.type ) {
    case TOGGLE_STATS:
      return {
        ...state,
        statsOn: !state.statsOn,
      };
    case TOGGLE_COUNT:
      return {
        ...state,
        totalCountOn: !state.totalCountOn,
      };
    default: return state;
  }
}

// action creators
export function toggleStats() {
  return { type: TOGGLE_STATS };
}

export function toggleTotalCount() {
  return { type: TOGGLE_COUNT };
}
export function toggleStreaming(){
  console.log("Hello World");
  initialState.liveStreaming = !initialState.liveStreaming;
  mq.publish("settings/streaming",{result:initialState.liveStreaming});
  return {type:TOGGLE_STREAMING};
}