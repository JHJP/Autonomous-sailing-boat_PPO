
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;


public class MoveToGoalAgent : Agent
{
    
    [SerializeField] private Transform targetTransform;
    [SerializeField] private Transform WeatherVaneTransform;
    // for the easier visalization of success and fail
    [SerializeField] private Material winMaterial;
    [SerializeField] private Material loseMaterial;
    [SerializeField] private MeshRenderer[] floorMeshRenderer;

    Rigidbody agentRigidbody;
    public float steerPower = 500f;
    public float motorPower = 5f;

    public override void Initialize() {
        agentRigidbody = gameObject.GetComponent<Rigidbody>();

    }
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(transform.localPosition);
        sensor.AddObservation(transform.rotation);
        sensor.AddObservation(targetTransform.localPosition);
        sensor.AddObservation(WeatherVaneTransform.localPosition.x);
        sensor.AddObservation(WeatherVaneTransform.localPosition.z);
    }
    public override void OnEpisodeBegin()
    {
        transform.localPosition = Vector3.zero;
        transform.rotation = Quaternion.identity;
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        int moveSteer = actions.DiscreteActions[0];
        int moveMotor = actions.DiscreteActions[1];

        float addForceToSteer = 0f;
        float addForceToMotor = 0f;

        switch (moveSteer) {
            case 0: addForceToSteer = 0f; break;
            case 1: addForceToSteer = 1f; break;
            case 2: addForceToSteer = -1f; break;
        }
        switch (moveMotor) {
            case 0: addForceToMotor = 0f; break;
            case 1: addForceToMotor = 1f; break;
            case 2: addForceToMotor = -1f; break;
        }
        // float moveSteer = actions.ContinuousActions[0];
        // float moveMotor = actions.ContinuousActions[1];
        agentRigidbody.AddTorque(0f, addForceToSteer*steerPower*Time.deltaTime, 0f);
		agentRigidbody.AddForce(transform.forward*addForceToMotor*motorPower*Time.deltaTime);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {

        // ActionSegment<float> continuousActions = actionsOut.ContinuousActions;
        // continuousActions[0] = Input.GetAxisRaw("Horizontal");
        // continuousActions[1] = Input.GetAxisRaw("Vertical");
        ActionSegment<int> discreteActions = actionsOut.DiscreteActions;
        switch (Mathf.RoundToInt(Input.GetAxisRaw("Horizontal"))) {
            case 0: discreteActions[0] = 0; break;
            case 1: discreteActions[0] = 1; break;
            case -1: discreteActions[0] = 2; break;
        }
        switch (Mathf.RoundToInt(Input.GetAxisRaw("Vertical"))) {
            case 0: discreteActions[1] = 0; break;
            case 1: discreteActions[1] = 1; break;
            case -1: discreteActions[1] = 2; break;
        }
        
    }

    public void OnTriggerEnter(Collider other) {
        if (other.TryGetComponent<Goal>(out Goal goal)) {
            SetReward(+1f);
            for(int i = 0; i < floorMeshRenderer.Length; i++) {
                floorMeshRenderer[i].material = winMaterial;
            }
            EndEpisode();
        }
        if (other.TryGetComponent<Wall>(out Wall wall)) {
            SetReward(-1f);
            for(int i = 0; i < floorMeshRenderer.Length; i++) {
                floorMeshRenderer[i].material = loseMaterial;
            }
            EndEpisode();
        }
    }
}
