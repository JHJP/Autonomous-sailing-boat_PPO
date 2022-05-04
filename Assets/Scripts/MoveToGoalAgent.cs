
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;


public class MoveToGoalAgent : Agent
{
    
    [SerializeField] private Transform targetTransform;
    // for the easier visalization of success and fail
    [SerializeField] private Material winMaterial;
    [SerializeField] private Material loseMaterial;
    [SerializeField] private MeshRenderer[] floorMeshRenderer;

    Rigidbody agentRigidbody;
    public float SteerPower = 500f;
    public float Power = 5f;

    public override void Initialize() {
        agentRigidbody = gameObject.GetComponent<Rigidbody>();

    }
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(transform.localPosition);
        sensor.AddObservation(transform.rotation);
        sensor.AddObservation(targetTransform.localPosition);
    }
    public override void OnEpisodeBegin()
    {
        transform.localPosition = Vector3.zero;
        transform.rotation = Quaternion.identity;
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        float moveSteer = actions.ContinuousActions[0];
        float moveMotor = actions.ContinuousActions[1];
        agentRigidbody.AddTorque(0f, moveSteer*SteerPower*Time.deltaTime, 0f);
		agentRigidbody.AddForce(transform.forward*moveMotor*Power*Time.deltaTime);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        ActionSegment<float> continuousActions = actionsOut.ContinuousActions;
        continuousActions[0] = Input.GetAxisRaw("Horizontal");
        continuousActions[1] = Input.GetAxisRaw("Vertical");
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
