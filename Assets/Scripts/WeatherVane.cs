using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WeatherVane : MonoBehaviour
{   
    public GameObject windEffect;
    Vector3 windDirection;
    Vector3 targetDirection;
    [SerializeField] private Transform vane;
    [SerializeField] private Transform baseTransform;
  
    float angle;
    void start() {
        windDirection = windEffect.GetComponent<WindEffect>().randomDirection;
    } 

    void Update() {
        targetDirection = vane.localPosition - baseTransform.localPosition;
        float angle = Vector3.SignedAngle(windDirection, targetDirection, Vector3.up);
        directionUpdate();
        transform.localPosition = Quaternion.Euler(0*Time.deltaTime, angle*Time.deltaTime, 0*Time.deltaTime) * transform.localPosition;
    }

    void directionUpdate() {
        windDirection = windEffect.GetComponent<WindEffect>().randomDirection;
    }

    private void OnDrawGizmos() {
        Gizmos.color = Color.magenta;
        Gizmos.DrawLine(baseTransform.position, transform.position);
    }
}
